import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
class SymbolBlock(HybridBlock):
    """Construct block from symbol. This is useful for using pre-trained models
    as feature extractors. For example, you may want to extract the output
    from fc2 layer in AlexNet.

    Parameters
    ----------
    outputs : Symbol or list of Symbol
        The desired output for SymbolBlock.
    inputs : Symbol or list of Symbol
        The Variables in output's argument that should be used as inputs.
    params : ParameterDict
        Parameter dictionary for arguments and auxililary states of outputs
        that are not inputs.

    Examples
    --------
    >>> # To extract the feature from fc1 and fc2 layers of AlexNet:
    >>> alexnet = gluon.model_zoo.vision.alexnet(pretrained=True, ctx=mx.cpu(),
                                                 prefix='model_')
    >>> inputs = mx.sym.var('data')
    >>> out = alexnet(inputs)
    >>> internals = out.get_internals()
    >>> print(internals.list_outputs())
    ['data', ..., 'model_dense0_relu_fwd_output', ..., 'model_dense1_relu_fwd_output', ...]
    >>> outputs = [internals['model_dense0_relu_fwd_output'],
                   internals['model_dense1_relu_fwd_output']]
    >>> # Create SymbolBlock that shares parameters with alexnet
    >>> feat_model = gluon.SymbolBlock(outputs, inputs, params=alexnet.collect_params())
    >>> x = mx.nd.random.normal(shape=(16, 3, 224, 224))
    >>> print(feat_model(x))
    """

    @staticmethod
    def imports(symbol_file, input_names, param_file=None, ctx=None, allow_missing=False, ignore_extra=False):
        """Import model previously saved by `gluon.HybridBlock.export` or
        `Module.save_checkpoint` as a `gluon.SymbolBlock` for use in Gluon.

        Parameters
        ----------
        symbol_file : str
            Path to symbol file.
        input_names : list of str
            List of input variable names
        param_file : str, optional
            Path to parameter file.
        ctx : Context, default None
            The context to initialize `gluon.SymbolBlock` on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.

        Returns
        -------
        gluon.SymbolBlock
            `gluon.SymbolBlock` loaded from symbol and parameter files.

        Examples
        --------
        >>> net1 = gluon.model_zoo.vision.resnet18_v1(
        ...     prefix='resnet', pretrained=True)
        >>> net1.hybridize()
        >>> x = mx.nd.random.normal(shape=(1, 3, 32, 32))
        >>> out1 = net1(x)
        >>> net1.export('net1', epoch=1)
        >>>
        >>> net2 = gluon.SymbolBlock.imports(
        ...     'net1-symbol.json', ['data'], 'net1-0001.params')
        >>> out2 = net2(x)
        """
        if is_np_array():
            sym = np_symbol.load(symbol_file)
        else:
            sym = symbol.load(symbol_file)
        if isinstance(input_names, str):
            input_names = [input_names]
        if param_file is None:
            inputs = [symbol.var(i, dtype=mx_real_t) for i in input_names]
        else:
            inputs = [symbol.var(i).as_np_ndarray() if is_np_array() else symbol.var(i) for i in input_names]
        ret = SymbolBlock(sym, inputs)
        if param_file is not None:
            ret.collect_params().load(param_file, ctx, allow_missing, ignore_extra, cast_dtype=True, dtype_source='saved')
        return ret

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['{block} : {numinputs} -> {numoutputs}'.format(block=self._cached_graph[1], numinputs=len(self._cached_graph[0]), numoutputs=len(self._cached_graph[1].list_outputs()))])
        return s.format(name=self.__class__.__name__, modstr=modstr)

    def __init__(self, outputs, inputs, params=None):
        super(SymbolBlock, self).__init__(prefix=None, params=None)
        self._prefix = ''
        self._params = ParameterDict('', params)
        if isinstance(inputs, symbol.Symbol) and len(inputs.list_outputs()) == 1:
            inputs = [inputs]
        if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
            outputs = outputs[0]
        syms, self._in_format = _flatten(inputs, 'input')
        out, self._out_format = _flatten(outputs, 'output')
        input_names = set()
        for i in syms:
            assert len(i.get_internals().list_outputs()) == 1, 'Input symbols must be variable, but %s is an output of operators' % str(i)
            input_names.add(i.name)
        row_sparse_storage = ndarray.ndarray._STORAGE_TYPE_STR_TO_ID['row_sparse']
        for i in out:
            for j in i.get_internals():
                assert j.attr('__storage_type__') != str(row_sparse_storage), "SymbolBlock doesn't support Parameter '%s' because its storage type is 'row_sparse'." % j.name
        if len(out) > 1:
            out = symbol.Group(out, _check_same_symbol_type(out))
        else:
            out = out[0]
        arg_params = out.list_arguments()
        aux_params = out.list_auxiliary_states()
        arg_types, aux_types = _infer_param_types(syms, out, arg_params, aux_params)
        for i, arg in enumerate(arg_params):
            if arg not in input_names:
                self.params.get(arg, allow_deferred_init=True, dtype=arg_types[i])
        for i, aux in enumerate(aux_params):
            if aux not in input_names:
                self.params.get(aux, grad_req='null', allow_deferred_init=True, dtype=aux_types[i])
        self._cached_graph = (syms, out)
        len_prefix = len(_common_prefix(list(self._params.keys())))
        self._reg_params = {key[len_prefix:]: val for key, val in self._params.items()}

    def forward(self, x, *args):
        if isinstance(x, NDArray):
            with x.ctx:
                return self._call_cached_op(x, *args)
        assert isinstance(x, Symbol), 'HybridBlock requires the first argument to forward be either Symbol or NDArray, but got %s' % type(x)
        args, in_fmt = _flatten([x] + list(args), 'input')
        assert in_fmt == self._in_format, 'Invalid input format'
        ret = copy.copy(self._cached_graph[1])
        ret._compose(**{k.name: v for k, v in zip(self._cached_graph[0], args)})
        return _regroup(list(ret), self._out_format)

    def _clear_cached_op(self):
        tmp = self._cached_graph
        super(SymbolBlock, self)._clear_cached_op()
        self._cached_graph = tmp

    def cast(self, dtype):
        self._clear_cached_op()
        super(SymbolBlock, self).cast(dtype)
        if np.dtype(dtype).name == 'float16':
            out = self._cached_graph[1]
            params_list = out.get_internals().list_inputs()
            for node in params_list:
                if node.endswith('running_var'):
                    prefix = node[:-11]
                    sibs = [prefix + t for t in ('running_mean', 'gamma', 'beta')]
                    is_bn = all((p in params_list for p in sibs))
                    if is_bn:
                        self.params.get(node).cast('float32')
                        for sib in sibs:
                            self.params.get(sib).cast('float32')
                if node.endswith('moving_var'):
                    prefix = node[:-10]
                    sibs = [prefix + t for t in ('moving_mean', 'gamma', 'beta')]
                    is_bn = all((p in params_list for p in sibs))
                    if is_bn:
                        self.params.get(node).cast('float32')
                        for sib in sibs:
                            self.params.get(sib).cast('float32')

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError

    def reset_ctx(self, ctx):
        """Re-assign all Parameters to other contexts. If the Block is hybridized, it will reset the _cached_op_args.
        Parameters
        ----------
        ctx : Context or list of Context, default :py:meth:`context.current_context()`.
            Assign Parameter to given context. If ctx is a list of Context, a
            copy will be made for each context.
        """
        params = self.collect_params()
        if self._cached_op:
            for p in self._cached_op_args:
                if p.name not in params:
                    p.reset_ctx(ctx)
        for p in params.values():
            p.reset_ctx(ctx)