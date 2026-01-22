import functools
import inspect
import operator
import types
from typing import Dict, List
import sympy
import torch._numpy as tnp
import torch.fx
import torch.random
from torch._dynamo import compiled_autograd
from torch.fx.experimental.symbolic_shapes import (
from .. import config, variables
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from ..utils import (
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import SizeVariable
class TensorVariable(VariableTracker):
    """A torch.Tensor input or an intermediate value in the FX graph"""
    _nonvar_fields = {'proxy', 'dtype', 'device', 'layout', 'ndim', 'size', 'stride', 'requires_grad', 'is_quantized', 'is_contiguous', 'is_sparse', 'class_type', 'specialized_value', *VariableTracker._nonvar_fields}

    def get_real_value(self):
        """
        Get the actual value represented by this variable if computation is run
        using the user-provided inputs.
        NOTE: this runs actual tensor computation and may be
        slow and memory-intensive.
        """
        return get_real_value(self.proxy.node, self.proxy.tracer)

    def __init__(self, proxy: torch.fx.Proxy, *, dtype, device, layout, ndim, requires_grad, is_quantized, is_sparse, class_type, size=None, stride=None, is_contiguous=None, specialized_value=None, **kwargs):
        super().__init__(**kwargs)
        self.proxy = proxy
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.ndim = ndim
        self.size = size
        self.stride = stride
        self.requires_grad = requires_grad
        self.is_quantized = is_quantized
        self.is_contiguous = is_contiguous
        self.is_sparse = is_sparse
        self.class_type = class_type
        self.specialized_value = specialized_value

    def as_proxy(self):
        return self.proxy

    def python_type(self):
        return self.class_type

    @staticmethod
    def specialize(value: torch.Tensor):
        props = {'dtype': value.dtype, 'device': value.device, 'layout': value.layout, 'ndim': int(value.ndim), 'requires_grad': value.requires_grad, 'is_quantized': value.is_quantized, 'is_sparse': value.is_sparse, 'class_type': type(value)}
        if not has_free_symbols(value):
            props['size'] = tuple([int(s) if is_symbolic(s) else s for s in value.size()])
            props['stride'] = tuple(value.stride())
            props['is_contiguous'] = tuple([x for x in torch._prims_common._memory_formats if value.is_contiguous(memory_format=x)])
        return props

    def dynamic_getattr(self, tx, name):
        if not self.source:
            raise NotImplementedError()
        scope = {'L': tx.output.local_scope, 'G': tx.output.global_scope}
        try:
            _input_associated_real_value = eval(self.source.name(), scope)
        except Exception as exc:
            raise NotImplementedError() from exc
        if _input_associated_real_value is None:
            raise NotImplementedError()
        if object_has_getattribute(_input_associated_real_value):
            raise NotImplementedError()
        if get_custom_getattr(_input_associated_real_value):
            raise NotImplementedError()
        real_value = getattr(_input_associated_real_value, name)
        if callable(real_value):
            raise NotImplementedError()
        from ..guards import GuardBuilder
        from .builder import VariableBuilder
        attr_source = AttrSource(self.source, name)
        install_guard(attr_source.make_guard(GuardBuilder.HASATTR))
        return VariableBuilder(tx, attr_source)(real_value)

    def var_getattr(self, tx, name):
        from . import ConstantVariable, TorchVariable
        if tx.strict_checks_enabled:
            if name in self._strict_mode_banned_ops():
                unimplemented(f'Illegal getattr invocation {name} in strict mode')
        result = None
        if name == 'ndim' and self.ndim is not None:
            result = ConstantVariable.create(self.ndim)
        elif name == 'dtype' and self.dtype is not None:
            result = ConstantVariable.create(self.dtype)
        elif name == 'device' and self.device is not None:
            result = ConstantVariable.create(self.device)
        elif name == 'layout' and self.layout is not None:
            result = TorchVariable(self.layout)
        elif name == 'is_cuda' and self.device is not None:
            result = ConstantVariable.create(self.device.type == 'cuda')
        elif name == 'shape' and self.size is not None:
            sizes = [variables.ConstantVariable.create(x) for x in self.size]
            result = SizeVariable(sizes)
        elif name == 'requires_grad' and self.requires_grad is not None:
            result = ConstantVariable.create(self.requires_grad)
        elif name == 'is_quantized' and self.is_quantized is not None:
            result = ConstantVariable.create(self.is_quantized)
        elif name == 'is_sparse' and self.is_sparse is not None:
            result = ConstantVariable.create(self.is_sparse)
        elif name == 'shape' and self.size is None:
            result = self.call_method(tx, 'size', [], {})
        elif name == 'ndim' and self.ndim is None:
            result = self.call_method(tx, 'dim', [], {})
        elif name == 'data':
            result = self.call_method(tx, 'detach', [], {})
        if name == '__class__':
            return TorchVariable(self.python_type())
        if result is not None and self.source is not None:
            install_guard(self.make_guard(GuardBuilder.TYPE_MATCH))
        if self.source is not None and hasattr(torch.ops.aten, name):
            fn = getattr(torch.ops.aten, name)
            if hasattr(fn, 'overloads') and hasattr(fn, fn.overloads()[0]) and (torch.Tag.inplace_view in getattr(fn, fn.overloads()[0]).tags):
                return variables.misc.DelayGraphBreakVariable()
        if result is None:

            def try_generic_attr_handling():
                from .builder import wrap_fx_proxy
                from .misc import GetAttrVariable
                try:
                    static_attr = inspect.getattr_static(torch.Tensor, name)
                except AttributeError:
                    return None
                if type(static_attr) != types.GetSetDescriptorType:
                    return None
                return wrap_fx_proxy(tx=tx, proxy=GetAttrVariable.create_getattr_proxy(self.as_proxy(), name))
            result = try_generic_attr_handling()
        if result is None:
            result = self.dynamic_getattr(tx, name)
        if result is None:
            raise NotImplementedError()
        return result

    def has_unpack_var_sequence(self, tx):
        return self.ndim > 0

    def unpack_var_sequence(self, tx, idxes=None):
        from .builder import wrap_fx_proxy_cls
        if idxes is None:
            if self.size:
                length = self.size[0]
            else:
                dyn_length = self.call_method(tx, 'size', [ConstantVariable.create(0)], {})
                assert isinstance(dyn_length, (SymNodeVariable, ConstantVariable))
                if isinstance(dyn_length, SymNodeVariable):
                    length = dyn_length.evaluate_expr(tx.output)
                else:
                    length = dyn_length.value
            idxes = range(length)
        return [wrap_fx_proxy_cls(target_cls=type(self), tx=tx, proxy=self.as_proxy()[i]) for i in idxes]

    def _strict_mode_banned_ops(self):
        return torch._dynamo.config._autograd_backward_strict_mode_banned_ops

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if tx.strict_checks_enabled:
            if name in self._strict_mode_banned_ops():
                unimplemented(f'Illegal method invocation {name} in strict mode')
        from . import ConstantVariable, TorchVariable, TupleVariable
        from .builder import wrap_fx_proxy
        from .user_defined import UserDefinedClassVariable
        kwargs = dict(kwargs)
        if name in ('stride', 'size'):
            dim_var = None
            if len(args) == 1:
                dim_var = args[0]
            elif 'dim' in kwargs:
                dim_var = kwargs['dim']
            else:
                assert not args and (not kwargs), f'Tensor.{name}() unhandled args/kwargs'
            dim = guard_if_dyn(dim_var)

            def make_const_size_variable(x, **options):
                return SizeVariable([ConstantVariable.create(y, **options) for y in x], **options)
            RetVariable = make_const_size_variable if name == 'size' else ConstantVariable.create
            if (r := getattr(self, name)) is not None:
                if dim is None:
                    return RetVariable(r)
                else:
                    return ConstantVariable.create(r[dim])
            if (fake := self.proxy.node.meta.get('example_value')) is not None:
                if dim is None:
                    fake_r = getattr(fake, name)()
                    if not has_free_symbols(fake_r):
                        return RetVariable(tuple((int(r) for r in fake_r)))
                else:
                    fake_r = getattr(fake, name)(dim)
                    if not has_free_symbols(fake_r):
                        return ConstantVariable.create(int(fake_r))
            return wrap_fx_proxy(tx, tx.output.create_proxy('call_method', name, *proxy_args_kwargs([self] + list(args), kwargs)))
        elif name in ('numel', 'nelement'):
            if self.size is not None:
                return ConstantVariable.create(product(self.size))
            if (fake := self.proxy.node.meta.get('example_value')) is not None:
                fake_r = fake.numel()
                if not has_free_symbols(fake_r):
                    return ConstantVariable.create(int(fake_r))
            assert not kwargs, f'Tensor.{name}() unhandled kwargs'
            return wrap_fx_proxy(tx, tx.output.create_proxy('call_method', 'numel', *proxy_args_kwargs([self] + list(args), kwargs)))
        elif name in ('ndimension', 'dim') and self.ndim is not None:
            constant_result = ConstantVariable.create(self.ndim)
        elif name == 'is_floating_point' and self.dtype is not None:
            constant_result = ConstantVariable.create(self.dtype.is_floating_point)
        elif name == 'is_contiguous':
            memory_format = kwargs.pop('memory_format').as_python_constant() if 'memory_format' in kwargs else torch.contiguous_format
            if self.is_contiguous is not None:
                constant_result = ConstantVariable.create(memory_format in self.is_contiguous)
            elif (fake := self.proxy.node.meta.get('example_value')) is not None:
                constant_result = ConstantVariable.create(fake.is_contiguous(memory_format=memory_format))
            else:
                constant_result = None
        elif name == 'type' and self.dtype is not None and (len(args) == 0) and isinstance(self.device, torch.device):
            tensortype = next((k for k, v in tensortype_to_dtype.items() if self.dtype in v))
            if self.device.type == 'cuda':
                constant_result = ConstantVariable.create(f'torch.cuda.{tensortype.__name__}')
            else:
                constant_result = ConstantVariable.create(f'torch.{tensortype.__name__}')
        elif name == 'type' and len(args) == 1 and (fqn(type(args[0].as_python_constant())) == 'torch.tensortype'):
            tensor_type = args[0].as_python_constant()
            tensor_type_const = ConstantVariable.create(fqn(tensor_type))
            return wrap_fx_proxy(tx, tx.output.create_proxy('call_method', name, *proxy_args_kwargs([self, tensor_type_const], kwargs)))
        elif name == 'as_subclass' and len(args) == 1 and isinstance(args[0], UserDefinedClassVariable):
            from .builder import VariableBuilder
            from .torch_function import TensorWithTFOverrideVariable
            py_cls = args[0].as_python_constant()
            torch_fn = VariableBuilder(tx, AttrSource(AttrSource(args[0].source, '__torch_function__'), '__func__'))(py_cls.__torch_function__.__func__)
            return TensorWithTFOverrideVariable.from_tensor_var(tx, self, py_cls, torch_fn)
        elif name == 'get_device' and isinstance(self.device, torch.device):
            index = self.device.index if self.device.type != 'cpu' else -1
            constant_result = ConstantVariable.create(index)
        else:
            constant_result = None
        if constant_result:
            assert not kwargs, f'Tensor.{name}() unhandled kwargs'
            if len(args) == 1:
                return constant_result.getitem_const(args[0])
            elif args:
                return TupleVariable([constant_result.getitem_const(a) for a in args])
            return constant_result
        elif name == 'numpy':
            if not config.trace_numpy:
                unimplemented('Tensor.numpy(). config.trace_numpy is False')
            if not np:
                unimplemented('Tensor.numpy(). NumPy is not available')
            assert not args, "Tensor.numpy() doesn't take args."
            if self.layout != torch.strided:
                raise TypeError(f"can't convert {self.layout} layout tensor to numpy. Use Tensor.dense() first")
            force = 'force' in kwargs and kwargs['force'].as_python_constant()
            if force:
                t = self.call_method(tx, 'detach', [], {})
                proxy = tx.output.create_proxy('call_method', 'cpu', (t.as_proxy(),), {})
            else:
                proxy = tx.output.create_proxy('call_method', 'view_as', *proxy_args_kwargs([self, self], {}))
            return NumpyNdarrayVariable.create(tx, proxy)
        elif name == 'tolist':
            from .builder import SourcelessBuilder

            def tolist(tensor, sub_proxy):

                def wrap(i, sub_proxy):
                    return SymNodeVariable.create(tx, sub_proxy.item(), sym_num=tx.output.shape_env.create_unbacked_symint())
                if tensor.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64]:
                    unimplemented('Input tensor for tolist must be an integer tensor')
                if tensor.dim() == 0:
                    return wrap(tensor, sub_proxy)
                if tensor.dim() == 1:
                    return [wrap(val, sub_proxy[i]) for i, val in enumerate(tensor)]
                return [tolist(sub_tensor, sub_proxy=sub_proxy[i]) for i, sub_tensor in enumerate(tensor)]
            tensor = self.as_proxy().node.meta['example_value']
            out = tolist(tensor, self.as_proxy())
            return SourcelessBuilder()(tx, out)
        elif name in ('backward', 'data_ptr'):
            unimplemented(f'Tensor.{name}')
        elif name == 'item' and (not config.capture_scalar_outputs):
            unimplemented(f'Tensor.{name}')
        elif name == '__len__':
            return self.call_method(tx, 'size', [ConstantVariable.create(0)], {})
        elif name == '__setitem__':
            key, value = args

            def has_bool_key(v):
                if isinstance(v, TensorVariable):
                    return v.dtype in (torch.bool, torch.int8)
                elif isinstance(v, TupleVariable):
                    return any((has_bool_key(item) for item in v.items))
                else:
                    return False
            if has_bool_key(key) and isinstance(value, TensorVariable) and value.requires_grad:
                unimplemented('boolean masking setitem backwards, see https://github.com/pytorch/pytorch/issues/114123')
            tx.output.create_proxy('call_function', operator.setitem, *proxy_args_kwargs([self] + list(args), kwargs))
            return ConstantVariable.create(None)
        elif name in ('resize_', 'resize_as_'):
            unimplemented(f'Tensor.{name}')
        elif name == 'set_' and len(args) > 1:
            unimplemented('Tensor.set_.source_Tensor_storage_offset')
        elif name == 'add_' and len(args) == 1 and (len(kwargs) == 1) and ('alpha' in kwargs):
            result = TorchVariable(torch.mul).call_function(tx, args + [kwargs['alpha']], {})
            return self.call_method(tx, 'add_', [result], {})
        elif name == 'addcdiv_' and len(args) == 2 and (len(kwargs) == 1) and ('value' in kwargs):
            result = TorchVariable(torch.div).call_function(tx, args, {})
            result = TorchVariable(torch.mul).call_function(tx, [result, kwargs['value']], {})
            return self.call_method(tx, 'add_', [result], {})
        elif name == '__contains__':
            result = TorchVariable(torch.eq).call_function(tx, [self, args[0]], {})
            result = TorchVariable(torch.any).call_function(tx, [result], {})
            return result.call_method(tx, 'item', [], {})
        elif name == 'redistribute':
            args_as_value = [x.as_python_constant() for x in args]
            kwargs_as_value = {k: v.as_python_constant() for k, v in kwargs.items()}

            def redistribute_fn_with_prim_types(x):
                return x.redistribute(*args_as_value, **kwargs_as_value)
            redistribute_fn_with_prim_types.__name__ = f'prim_{name}'
            return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', redistribute_fn_with_prim_types, *proxy_args_kwargs([self], {})))
        elif name in {'register_hook', 'register_post_accumulate_grad_hook'}:
            assert len(args) == 1
            fn_var = args[0]
            if not isinstance(fn_var, (variables.functions.FunctoolsPartialVariable, variables.UserFunctionVariable, variables.TorchVariable, variables.NNModuleVariable)):
                unimplemented('Unexpected callable type passed to register_hook')
            if isinstance(fn_var, variables.NestedUserFunctionVariable):
                unimplemented('NYI - lambda variables as hooks')
            elif isinstance(fn_var, variables.functions.FunctoolsPartialVariable):
                fn = fn_var.as_python_constant()
            else:
                fn = fn_var.fn
            handle_variable = variables.user_defined.RemovableHandleVariable(mutable_local=variables.base.MutableLocal())
            if not self.source:
                src = fn_var.source
                if not src and isinstance(fn_var, variables.functions.FunctoolsPartialVariable) and fn_var.func.source:
                    src = fn_var.func.source
                if not src:
                    unimplemented('No source for register_hook target fn')
                tx.output.guards.add(src.make_guard(GuardBuilder.ID_MATCH))
                if not compiled_autograd.compiled_autograd_enabled:
                    unimplemented('Compilation of intermediate hooks requires compiled autograd')
                fn = functools.partial(trace_wrapped, fn=fn)

                def _register_hook_trampoline(tensor):
                    hook_callable = getattr(tensor, name)
                    hook_callable(fn)
                    return tensor
                return wrap_fx_proxy(tx, tx.output.create_proxy('call_function', _register_hook_trampoline, (self.as_proxy(),), {}))
            tx.output.side_effects.register_hook(self, fn_var, handle_variable, name)
            return handle_variable
        elif name == 'requires_grad_' and self.as_proxy().node.meta['example_value'].requires_grad != (args[0].value if len(args) > 0 else True):
            unimplemented('Tensor.requires_grad_')
        else:
            if name == 'new' and len(args) == 1 and isinstance(args[0], SizeVariable):
                name = 'new_empty'
            return wrap_fx_proxy(tx, tx.output.create_proxy('call_method', name, *proxy_args_kwargs([self] + list(args), kwargs)))

    def rename(self, tx, name):
        self.proxy.node._rename(name)
        return super().rename(tx, name)