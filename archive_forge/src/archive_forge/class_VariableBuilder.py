import abc
import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import logging
import operator
import re
import sys
import types
from typing import List, NamedTuple, Optional, Union
import torch
from torch import SymInt
from torch._guards import GuardSource, TracingContext
from torch._ops import HigherOrderOperator
from torch._streambase import _EventBase, _StreamBase
from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.immutable_collections import immutable_list
from torch.nested._internal.nested_tensor import NestedTensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.weak import TensorWeakRef
from .. import config, mutation_guard, replay_record, skipfiles, trace_rules
from ..allowed_functions import (
from ..device_interface import get_registered_device_interfaces
from ..exc import InternalTorchDynamoError, unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..side_effects import SideEffects
from ..source import (
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
from .dicts import (
from .distributed import (
from .functions import (
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lazy import LazyVariableTracker
from .lists import (
from .misc import (
from .nn_module import FSDPManagedNNModuleVariable, UnspecializedNNModuleVariable
from .optimizer import OptimizerVariable
from .tensor import (
from .torch import torch_special_class_types, TorchVariable
from .torch_function import build_torch_function_fn, TensorWithTFOverrideVariable
from .user_defined import (
class VariableBuilder:
    """Wrap a python value in a VariableTracker() instance"""

    def __init__(self, tx, source: Source):
        assert source is not None, 'Consider SourcelessBuilder for ephemeral objects, usually objects created locally.'
        assert TracingContext.try_get() is not None, 'Expected active TracingContext'
        super().__init__()
        self.tx = tx
        self.source = source
        self.name = source.name()

    def __call__(self, value):
        if value in self.tx.output.side_effects:
            side_effect_result = self.tx.output.side_effects[value]
            dup_guard = make_dupe_guard(self.source, side_effect_result.source)
            if dup_guard:
                self.install_guards(dup_guard)
            return side_effect_result
        vt = self._wrap(value).clone(**self.options())
        if self._can_lift_attrs_to_inputs(vt):
            vt = self.tx.output.side_effects.track_object_existing(self.source, value, vt)
        return vt

    def _can_lift_attrs_to_inputs(self, vt):
        if type(vt) in [TensorVariable, TensorWithTFOverrideVariable, UserDefinedObjectVariable, NumpyNdarrayVariable]:
            return True
        return False

    @staticmethod
    @functools.lru_cache(None)
    def _common_constants():
        return {0, 1}

    def get_source(self):
        return self.source

    def options(self):
        return {'source': self.get_source()}

    def install_guards(self, *guards):
        source = self.get_source()
        if isinstance(source, ConstantSource) or source.guard_source() == GuardSource.CONSTANT:
            return None
        install_guard(*[source.make_guard(guard) for guard in guards], skip=1)
        return {}

    @classmethod
    @functools.lru_cache(None)
    def _type_dispatch(cls):
        entries = [((torch.Tensor, torch.nn.Parameter, torch._subclasses.FakeTensor, torch._subclasses.functional_tensor.FunctionalTensor), cls.wrap_tensor), ((tuple, list, odict_values, collections.deque), cls.wrap_listlike), (tuple_iterator, cls.wrap_tuple_iterator), ((slice, range), cls.wrap_slice_range), ((int, float, bool, type(None), str, torch.Size, torch.device, torch.dtype), cls.wrap_literal)]
        if config.trace_numpy and np:
            entries.append((np.ndarray, cls.wrap_numpy_ndarray))
        result = {}
        for ts, fn in entries:
            for t in ts if isinstance(ts, tuple) else (ts,):
                assert t not in result
                result[t] = fn
        return result

    @classmethod
    @functools.lru_cache(None)
    def _id_dispatch(cls):
        from ..comptime import comptime
        entries = [(inspect.signature, lambda self, value: LambdaVariable(InspectSignatureVariable.create, source=self.source, **self.install_guards(GuardBuilder.CLOSURE_MATCH))), (comptime, lambda self, value: ComptimeVariable()), (dataclasses.fields, lambda self, value: LambdaVariable(_dataclasses_fields_lambda, source=self.source, **self.install_guards(GuardBuilder.FUNCTION_MATCH)))]
        result = {}
        for ts, fn in entries:
            for t in ts if isinstance(ts, (tuple, list)) else (ts,):
                assert t not in result
                result[id(t)] = fn
        return result

    def _wrap(self, value):
        from torch.utils._triton import has_triton
        if has_triton():
            from triton.runtime.autotuner import Autotuner
            from triton.runtime.jit import JITFunction
        else:

            class JITFunction:
                pass

            class Autotuner:
                pass
        type_dispatch = self._type_dispatch().get(type(value))
        if type_dispatch is not None:
            return type_dispatch(self, value)
        id_dispatch = self._id_dispatch().get(id(value))
        if id_dispatch is not None:
            return id_dispatch(self, value)
        value = inspect.getattr_static(value, '_torchdynamo_inline', value)
        if is_traceable_wrapper_subclass(value) or istype(value, config.traceable_tensor_subclasses):
            return self.wrap_tensor(value)
        elif is_namedtuple(value):
            return self.wrap_listlike(value)
        elif value is torch.utils._pytree.SUPPORTED_NODES:
            self.install_guards(GuardBuilder.DICT_VERSION)
            result = {k: UserDefinedObjectVariable(value[k], source=GetItemSource(self.get_source(), k)) for k in value.keys()}
            return ConstDictVariable(result, type(value))
        elif value is sys.modules:
            return PythonSysModulesVariable(source=self.source)
        elif istype(value, (dict, collections.defaultdict, collections.OrderedDict)) and all((ConstantVariable.is_literal(k) or self.tensor_can_be_dict_key(k) or isinstance(k, enum.Enum) for k in value.keys())):
            if not value and self.get_source().is_nn_module():
                self.install_guards(GuardBuilder.BOOL_FALSE)
            else:
                self.install_guards(GuardBuilder.DICT_KEYS)
            for key in value.keys():
                if self.tensor_can_be_dict_key(key):
                    self.tx.store_global_weakref(global_key_name(key), key)

            def index_source(key):
                if self.tensor_can_be_dict_key(key):
                    return GlobalWeakRefSource(global_key_name(key))
                else:
                    return key
            result = {k: LazyVariableTracker.create(value[k], source=GetItemSource(self.get_source(), index_source(k))) for k in value.keys()}
            if istype(value, collections.defaultdict):
                result = DefaultDictVariable(result, type(value), self._wrap(value.default_factory))
            else:
                result = ConstDictVariable(result, type(value))
            return self.tx.output.side_effects.track_dict(self.source, value, result)
        elif isinstance(value, torch.nn.Module):
            return self.wrap_module(value)
        elif ConstantVariable.is_literal(value):
            return self.wrap_literal(value)
        elif istype(value, frozenset) and all((is_allowed(x) or ConstantVariable.is_literal(x) for x in value)):
            self.install_guards(GuardBuilder.ID_MATCH)
            return ConstantVariable.create(value=value, source=self.source)
        elif isinstance(value, enum.Enum):
            self.install_guards(GuardBuilder.ID_MATCH)
            return EnumVariable(value=value, source=self.source)
        elif is_builtin_callable(value):
            self.install_guards(GuardBuilder.BUILTIN_MATCH)
            return BuiltinVariable(value, source=self.source)
        elif is_utils_checkpoint(value):
            return build_checkpoint_variable(source=self.source)
        elif isinstance(value, functools.partial):
            func_src = AttrSource(self.get_source(), 'func')
            func_obj = VariableBuilder(self.tx, func_src)(value.func)
            args = []
            args_source = AttrSource(self.get_source(), 'args')
            for i, arg in enumerate(value.args):
                args.append(VariableBuilder(self.tx, GetItemSource(args_source, i))(arg))
            keywords = {}
            keywords_source = AttrSource(self.get_source(), 'keywords')
            for k, v in value.keywords.items():
                keywords[k] = VariableBuilder(self.tx, GetItemSource(keywords_source, k))(v)
            install_guard(self.get_source().make_guard(GuardBuilder.TYPE_MATCH), keywords_source.make_guard(GuardBuilder.DICT_KEYS), args_source.make_guard(GuardBuilder.LIST_LENGTH))
            return FunctoolsPartialVariable(func_obj, args, keywords, original=value)
        elif is_typing(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return TypingVariable(value, source=self.source)
        elif np is not None and isinstance(value, np.generic):
            return self.wrap_numpy_ndarray(np.asarray(value))
        elif is_numpy(value):
            assert np
            self.install_guards(GuardBuilder.FUNCTION_MATCH if callable(value) else GuardBuilder.TYPE_MATCH)
            return NumpyVariable(value, source=self.source)
        elif CollectiveFunctionRewriteVariable.can_rewrite(value):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return CollectiveFunctionRewriteVariable.create(self.tx, value, source=self.source)
        elif istype(value, torch.autograd.function.FunctionMeta):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return AutogradFunctionVariable(value, source=self.source)
        elif isinstance(value, torch.autograd.function.FunctionCtx):
            saved_tensors_source = AttrSource(self.source, 'saved_tensors')
            install_guard(self.source.make_guard(GuardBuilder.TYPE_MATCH), saved_tensors_source.make_guard(GuardBuilder.LIST_LENGTH))
            saved_tensors = [VariableBuilder(self.tx, GetItemSource(saved_tensors_source, n))(v) for n, v in enumerate(value.saved_tensors)]
            return self.tx.output.side_effects.track_object_existing(self.source, value, AutogradFunctionContextVariable(value, source=self.source, saved_tensors=SavedTensorBox(saved_tensors)))
        elif isinstance(value, types.MethodType) and istype(getattr(value, '__self__', None), torch.autograd.function.FunctionMeta) and (getattr(value, '__name__', '') == 'apply') and (value == getattr(value.__self__, 'apply', None)):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return GetAttrVariable(AutogradFunctionVariable(value.__self__, source=self.source), 'apply')
        elif np and isinstance(value, np.number):
            return self.wrap_unspecialized_primitive(value)
        elif DataClassVariable.is_matching_object(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return DataClassVariable.wrap(self, value)
        elif HFPretrainedConfigVariable.is_matching_object(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return HFPretrainedConfigVariable(value)
        elif isinstance(value, HigherOrderOperator):
            self.install_guards(GuardBuilder.TYPE_MATCH, GuardBuilder.NAME_MATCH)
            return TorchHigherOrderOperatorVariable.make(value, source=self.source)
        elif type(value).__name__ == 'builtin_function_or_method' and isinstance(value.__self__, torch_special_class_types):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return TorchVariable(value)
        elif isinstance(value, _StreamBase):
            self.install_guards(GuardBuilder.ID_MATCH)
            return StreamVariable(None, value, value.device.type, source=self.source)
        elif isinstance(value, _EventBase):
            self.install_guards(GuardBuilder.ID_MATCH)
            return EventVariable(None, value, source=self.source)
        elif isinstance(value, torch._C._TensorMeta) and value in config.traceable_tensor_subclasses:
            return TensorSubclassVariable(value, source=self.source)
        elif istype(value, contextlib.nullcontext) and inspect.getattr_static(value, 'enter_result', None) is None:
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return NullContextVariable(source=self.source)
        elif KeyedJaggedTensorVariable.is_matching_object(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = KeyedJaggedTensorVariable(value, source=self.source)
            return self.tx.output.side_effects.track_object_existing(self.source, value, result)
        elif isinstance(value, torch.optim.Optimizer):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return OptimizerVariable(value, source=self.source)
        elif ProcessGroupVariable.is_process_group(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return ProcessGroupVariable(value, source=self.source)
        elif DeviceMeshVariable.is_device_mesh(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return DeviceMeshVariable(value, source=self.source)
        elif PlacementClassVariable.is_placement_type(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return PlacementClassVariable(value, source=self.source)
        elif PlacementVariable.is_placement(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return PlacementVariable(value, source=self.source)
        elif isinstance(value, torch.SymBool):
            value_hint = value.node.require_hint()
            new_source = ConvertIntSource(self.source)
            new_symint = self.tx.output.shape_env.create_unspecified_symint_and_symbol(int(value_hint), new_source, dynamic_dim=DimDynamic.DYNAMIC)
            sym_node_proxy = self.tx.output.root_tracer.create_graph_input(re.sub('[^a-zA-Z0-9]+', '_', self.name), type(new_symint), source=new_source)
            sym_node_proxy.node.meta['grapharg'] = GraphArg(new_source, new_symint, False, None, is_tensor=False, example_strong_ref=new_symint)
            self.tx.output.bound_symbols.add(new_symint.node.expr)
            self.tx.output.tracked_fakes.append(TrackedFake(new_symint, new_source, None))
            return SymNodeVariable(sym_node_proxy, new_symint == 1)
        elif isinstance(value, (JITFunction, Autotuner)):
            self.install_guards(GuardBuilder.ID_MATCH)
            return TritonKernelVariable(value, None, None, source=self.source)
        elif isinstance(value, torch.amp.autocast_mode.autocast):
            self.install_guards(GuardBuilder.ID_MATCH)
            return AutocastModeVariable(target_values=[value.device, value.fast_dtype, value._enabled, value._cache_enabled], source=self.source)
        elif trace_rules.lookup(value) is not None:
            if is_user_defined_allowed(value):
                self.tx.output.has_user_defined_allowed_in_graph = True
            return trace_rules.lookup(value).create_with_source(value, source=self.source)
        elif is_allowed(value):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return TorchVariable(value, source=self.source)
        elif istype(value, (type, types.FunctionType)) and skipfiles.check(value, is_inlined_call=True) and (not inspect.getattr_static(value, '_torchdynamo_inline', False)) and (not inspect.getattr_static(value, '__script_if_tracing_wrapper', False)):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return SkipFilesVariable(value, skipfiles.check_verbose(value, is_inlined_call=True).reason, source=self.source)
        elif istype(value, (types.FunctionType, torch.jit.ScriptFunction)):
            self.install_guards(GuardBuilder.CLOSURE_MATCH)
            return UserFunctionVariable(value, source=self.source)
        elif isinstance(value, types.MethodType) and isinstance(value.__self__, torch.nn.Module):
            self_obj = VariableBuilder(self.tx, source=AttrSource(self.source, '__self__'))(value.__self__)
            assert self_obj and isinstance(self_obj, VariableTracker), 'Failed to produce a valid self obj'
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return UserMethodVariable(value.__func__, self_obj, source=self.source)
        elif istype(value, (types.ModuleType, replay_record.DummyModule)):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return PythonModuleVariable(value, source=self.source)
        elif isinstance(value, types.GetSetDescriptorType):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return GetSetDescriptorVariable(value)
        elif isinstance(value, types.MethodWrapperType):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return MethodWrapperVariable(value, source=self.source)
        elif issubclass(type(value), type):
            self.install_guards(GuardBuilder.FUNCTION_MATCH)
            return UserDefinedClassVariable(value, source=self.source)
        elif RestrictedListSubclassVariable.is_matching_cls(type(value)):
            self.install_guards(GuardBuilder.TYPE_MATCH, GuardBuilder.LIST_LENGTH)
            return self.tx.output.side_effects.track_list(self.source, value, RestrictedListSubclassVariable([LazyVariableTracker.create(value=value[i], source=GetItemSource(self.source, i)) for i in range(len(value))], user_cls=type(value), user_cls_source=AttrSource(self.source, '__class__')))
        else:
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = UserDefinedObjectVariable(value, source=self.source)
            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                return result
            return self.tx.output.side_effects.track_object_existing(self.source, value, result)

    def tensor_can_be_dict_key(self, value):
        return isinstance(value, torch.nn.Parameter) or (isinstance(self.source, AttrSource) and self.source.member == 'state' and isinstance(self.source.base, LocalSource))

    def tensor_should_specialize(self):
        return self.source and isinstance(self.source, GetItemSource) and isinstance(self.source.base, GetItemSource) and (self.source.base.index == 'params') and isinstance(self.source.base.base, GetItemSource) and isinstance(self.source.base.base.base, AttrSource) and (self.source.base.base.base.member == 'param_groups') and isinstance(self.source.base.base.base.base, LocalSource) and (isinstance(self.tx.f_locals[self.source.base.base.base.base.local_name], torch.optim.Optimizer) if self.source.base.base.base.base.local_name in self.tx.f_locals.keys() else True)

    def wrap_listlike(self, value: Union[tuple, list, odict_values, NamedTuple]):
        self.install_guards(GuardBuilder.LIST_LENGTH)
        for item in value:
            if item is value:
                unimplemented('list elements are pointing to the list itself')
        output = [VariableBuilder(self.tx, GetItemSource(self.get_source(), i))(item) for i, item in enumerate(value)]
        result = BaseListVariable.cls_for_instance(value)(output, mutable_local=MutableLocal())
        if istype(value, list):
            return self.tx.output.side_effects.track_list(self.source, value, result)
        return result

    def wrap_tuple_iterator(self, value: tuple_iterator):
        self.install_guards(GuardBuilder.TUPLE_ITERATOR_LEN)
        output = [VariableBuilder(self.tx, TupleIteratorGetItemSource(self.get_source(), i))(tuple_iterator_getitem(value, i)) for i in range(tuple_iterator_len(value))]
        return TupleIteratorVariable(output, mutable_local=MutableLocal())

    def wrap_slice_range(self, value: Union[slice, range]):
        items = [VariableBuilder(self.tx, AttrSource(self.get_source(), k))(getattr(value, k)) for k in ('start', 'stop', 'step')]
        self.install_guards(GuardBuilder.TYPE_MATCH)
        if isinstance(value, slice):
            return SliceVariable(items)
        else:
            return RangeVariable(items)

    def wrap_module(self, value: torch.nn.Module):
        from ..eval_frame import OptimizedModule
        if istype(value, OptimizedModule):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.source = AttrSource(self.source, '_orig_mod')
            return self.wrap_module(value._orig_mod)
        if isinstance(value, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)) and (not config.allow_rnn):
            unimplemented('TorchDynamo purposely graph breaks on RNN, GRU, LSTMs')
        if mutation_guard.is_dynamic_nn_module(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = UnspecializedNNModuleVariable(value)
            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                return result
            return self.tx.output.side_effects.track_object_existing(self.source, value, result)
        elif issubclass(value.__class__, torch.nn.parallel.distributed.DistributedDataParallel):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return UnspecializedNNModuleVariable(value)
        elif getattr(value, '_is_fsdp_managed_module', False):
            assert getattr(value, '_fsdp_use_orig_params', False), 'Dynamo only supports FSDP with use_orig_params=True'
            self.install_guards(GuardBuilder.TYPE_MATCH, GuardBuilder.ID_MATCH)
            return FSDPManagedNNModuleVariable(value, source=self.get_source())
        else:
            return self.tx.output.register_attr_or_module(value, self.name, source=self.get_source())

    def wrap_literal(self, value):
        unspec = not config.specialize_int
        if unspec and type(value) is torch.Size:
            self.install_guards(GuardBuilder.LIST_LENGTH)
            return SizeVariable([VariableBuilder(self.tx, GetItemSource(self.get_source(), i))(v) for i, v in enumerate(value)])
        elif unspec and type(value) is int:
            if not TracingContext.get().force_unspec_int_unbacked_size_like and (value in self._common_constants() or not self.source.guard_source().is_local() or self.source.guard_source().is_nn_module()):
                self.install_guards(GuardBuilder.CONSTANT_MATCH)
                return ConstantVariable.create(value=value)
            else:
                return self.wrap_unspecialized_primitive(value)
        else:
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            return ConstantVariable.create(value=value)

    def assert_not_wrapped_by_this_graph(self, value: torch.Tensor):
        if is_fake(value) and maybe_get_fake_mode(value) is self.tx.fake_mode:
            raise InternalTorchDynamoError('Cannot wrap a Tensor that has already been', 'wrapped by this instance of Dynamo')

    def wrap_tensor(self, value: torch.Tensor):
        source = self.get_source()
        assert value not in self.tx.output.side_effects
        if (source.guard_source().is_nn_module() or get_static_address_type(value) is not None) and (not source.guard_source().is_fsdp_module()):
            self.assert_not_wrapped_by_this_graph(value)
            return self.tx.output.register_attr_or_module(value, self.name, source=source)
        if is_constant_source(source):
            self.assert_not_wrapped_by_this_graph(value)
            return self.tx.output.register_attr_or_module(value, re.sub('[^a-zA-Z0-9]+', '_', self.name), source=source)
        if type(value) in config.traceable_tensor_subclasses:
            subclass_type = type(value)
        else:
            assert type(value) in (torch.Tensor, torch.nn.Parameter, torch._subclasses.fake_tensor.FakeTensor, torch._subclasses.functional_tensor.FunctionalTensor) or is_traceable_wrapper_subclass(value), type(value)
            subclass_type = None
        is_duplicate_tensor = source in self.tx.output.input_source_to_var
        if is_duplicate_tensor:
            return self.tx.output.input_source_to_var[source]
        self.assert_not_wrapped_by_this_graph(value)
        tensor_proxy = self.tx.output.root_tracer.create_graph_input(re.sub('[^a-zA-Z0-9]+', '_', self.name), type(value), source=source)
        options = {}
        if type(value) in config.traceable_tensor_subclasses:
            options['torch_function_fn'] = build_torch_function_fn(self.tx, value, self.source)
            self.install_guards(GuardBuilder.TYPE_MATCH)
        if isinstance(value, torch.Tensor) and value.is_nested and (not isinstance(value, NestedTensor)):
            unimplemented('torch.compile does not support strided NestedTensor')
        tensor_variable = wrap_fx_proxy(tx=self.tx, proxy=tensor_proxy, example_value=value, should_specialize=self.tensor_should_specialize(), subclass_type=subclass_type, source=source, **options)
        self.install_guards(functools.partial(GuardBuilder.TENSOR_MATCH, value=value if isinstance(source, NumpyTensorSource) else TensorWeakRef(value)))
        self.tx.output.input_source_to_var[source] = tensor_variable
        assert 'tensor_dict' not in tensor_proxy.node.meta
        tensor_proxy.node.meta['tensor_dict'] = value.__dict__.copy()
        fake_tensor_value = tensor_variable.proxy.node.meta['example_value']
        if maybe_get_fake_mode(fake_tensor_value) is not self.tx.fake_mode:
            raise InternalTorchDynamoError("Wrapped Tensor must be this graph's fake")
        grapharg = GraphArg(source, value, False, fake_tensor_value)
        tensor_proxy.node.meta['grapharg'] = grapharg
        self.tx.output.add_symbol_bindings(grapharg)
        return tensor_variable

    def wrap_numpy_ndarray(self, value):
        assert np is not None
        assert isinstance(value, np.ndarray)
        source = NumpyTensorSource(self.get_source())
        from torch._numpy import _util
        readonly = not value.flags.writeable
        if readonly:
            value.flags.writeable = True
        try:
            tensor_value = _util._try_convert_to_tensor(value)
            if readonly:
                from torch._prims_common import clone_preserve_strides
                tensor_value = clone_preserve_strides(tensor_value)
        except NotImplementedError as e:
            unimplemented(str(e))
        VariableBuilder(self.tx, source)(tensor_value).recursive_realize()
        proxy = self.tx.output.root_tracer.create_graph_input(re.sub('[^a-zA-Z0-9]+', '_', self.name), type(tensor_value), source=source)
        options = {'source': source}
        numpy_ndarray_variable = wrap_fx_proxy_cls(target_cls=NumpyNdarrayVariable, tx=self.tx, proxy=proxy, example_value=tensor_value, **options)
        self.tx.output.input_source_to_var[source] = numpy_ndarray_variable
        example_value = numpy_ndarray_variable.proxy.node.meta['example_value']
        grapharg = GraphArg(source, tensor_value, is_unspecialized=True, fake_tensor=example_value, is_tensor=True, example_strong_ref=tensor_value)
        proxy.node.meta['grapharg'] = grapharg
        return numpy_ndarray_variable

    def wrap_unspecialized_primitive(self, value):
        if self.name in self.tx.output.unspec_variable_map:
            return self.tx.output.unspec_variable_map[self.name]
        else:
            shape_env = self.tx.output.shape_env
            if TracingContext.get().force_unspec_int_unbacked_size_like and isinstance(value, int):
                wrapped_value = shape_env.create_unbacked_symint()
                _constrain_range_for_size(wrapped_value)
                self.tx.output.bound_symbols.add(wrapped_value.node.expr)
                self.tx.output.tracked_fakes.append(TrackedFake(wrapped_value, self.source, None))
            elif isinstance(value, int) and (not is_constant_source(self.get_source())) and (not isinstance(self.get_source(), RandomValueSource)):
                if torch._dynamo.config.specialize_int:
                    self.install_guards(GuardBuilder.CONSTANT_MATCH)
                    return ConstantVariable.create(value=value)
                name = self.source.name()
                if name not in self.tx.output.frame_state:
                    frame_state_entry = FrameStateSizeEntry(scalar=value, size=None)
                else:
                    frame_state_entry = self.tx.output.frame_state[name]
                    if frame_state_entry.scalar != value:
                        log.debug('automatic dynamic int %s val %s != %s', name, value, frame_state_entry.scalar)
                        frame_state_entry.scalar = None
                self.tx.output.frame_state[name] = frame_state_entry
                if config.automatic_dynamic_shapes and frame_state_entry.scalar is None or not config.assume_static_by_default:
                    dynamic_dim = DimDynamic.DYNAMIC
                else:
                    self.install_guards(GuardBuilder.CONSTANT_MATCH)
                    return ConstantVariable.create(value=value)
                wrapped_value = shape_env.create_unspecified_symint_and_symbol(value, source=self.source, dynamic_dim=dynamic_dim)
                self.tx.output.bound_symbols.add(wrapped_value.node.expr)
                self.tx.output.tracked_fakes.append(TrackedFake(wrapped_value, self.source, None))
            else:
                wrapped_value = torch.tensor(value)
            if not isinstance(self.get_source(), RandomValueSource):
                install_guard(self.get_source().make_guard(GuardBuilder.TYPE_MATCH))
            options = {'source': self.get_source()}
            if isinstance(wrapped_value, torch.Tensor):
                options.update({'raw_value': value})
            proxy = self.tx.output.root_tracer.create_graph_input(re.sub('[^a-zA-Z0-9]+', '_', self.name), type(wrapped_value), source=self.get_source())
            unspec_var = wrap_fx_proxy_cls(UnspecializedPythonVariable, tx=self.tx, proxy=proxy, example_value=wrapped_value, **options)
            self.tx.output.unspec_variable_map[self.name] = unspec_var
            if not is_constant_source(self.get_source()):
                if self.tx.export and (not isinstance(self.get_source(), LocalSource)):
                    raise AssertionError('Dynamo attempts to add additional input during export: value={}, source={}'.format(wrapped_value, self.get_source()))
                fake_tensor_value = None
                if isinstance(unspec_var, ConstantVariable):
                    example_value = unspec_var.value
                else:
                    example_value = unspec_var.proxy.node.meta['example_value']
                if is_fake(example_value):
                    fake_tensor_value = example_value
                    assert fake_tensor_value.fake_mode is self.tx.fake_mode, f"fake mode ({fake_tensor_value.fake_mode}) from fake tensor metadata doesn't match mode({{self.tx.fake_mode}}) from InstructionTranslator"
                proxy.node.meta['grapharg'] = GraphArg(self.get_source(), wrapped_value, isinstance(wrapped_value, torch.Tensor), fake_tensor_value, is_tensor=False, example_strong_ref=wrapped_value)
            return unspec_var