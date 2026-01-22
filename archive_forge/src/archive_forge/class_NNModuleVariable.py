import functools
import inspect
import itertools
import types
from contextlib import contextmanager, nullcontext
from typing import Dict, List
import torch.nn
from .. import skipfiles, variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented, UnspecializeRestartAnalysis, Unsupported
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import GenerationTracker
from ..source import (
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .functions import invoke_and_store_as_constant
from .lists import SliceVariable
from .user_defined import UserDefinedObjectVariable
class NNModuleVariable(VariableTracker):
    _nonvar_fields = {'module_type', 'module_key', *VariableTracker._nonvar_fields}

    def __init__(self, module_type: type, module_key: str, **kwargs):
        super().__init__(**kwargs)
        self.module_type = module_type
        self.module_key = module_key
        assert self.source

    def python_type(self):
        return self.module_type

    def _wrap_submodule(self, tx, source, submod, *key_extra, **options):
        return

    def unpack_var_sequence(self, tx):
        base = tx.output.get_submodule(self.module_key)
        if isinstance(base, torch.nn.ModuleDict):
            result = []
            for name, submod in base.items():
                name_var = variables.ConstantVariable.create(name)
                tx.output.register_attr_or_module(submod, self.module_key, name, source=NNModuleSource(GetItemSource(self.source, name)))
                result.append(name_var)
            return result
        assert isinstance(base, (torch.nn.ModuleList, torch.nn.ParameterList, torch.nn.Sequential)), typestr(base)
        assert self.source
        result = []
        for idx, submod in enumerate(base):
            result.append(tx.output.register_attr_or_module(submod, self.module_key, idx, source=NNModuleSource(GetItemSource(self.source, idx))))
        return result

    def call_hasattr(self, tx, name: str) -> 'VariableTracker':
        mod = tx.output.get_submodule(self.module_key)
        result = hasattr(mod, name)
        install_guard(NNModuleSource(AttrSource(self.source, name)).make_guard(GuardBuilder.HASATTR))
        return variables.ConstantVariable.create(result)

    def is_training(self, tx):
        mod = tx.output.get_submodule(self.module_key)
        return getattr(mod, 'training', False)

    def convert_to_unspecialized(self, tx):
        """Restart analysis treating this module as an UnspecializedNNModuleVariable"""
        mod = tx.output.get_submodule(self.module_key)
        GenerationTracker.tag(mod)
        if tx.f_code.co_name != '__init__':
            GenerationTracker.mark_class_dynamic(type(mod))
        raise UnspecializeRestartAnalysis()

    def _custom_getattr_fallback(self, base, tx, name, options):
        """Check for a __getattr__ and handle it specially if it is implemented"""
        if object_has_getattribute(base):
            unimplemented('torch.nn.Module with a custom __getattribute__ defined')
        getattr_fn = get_custom_getattr(base)
        if getattr_fn is None:
            return None
        if not isinstance(getattr_fn, types.FunctionType):
            unimplemented('torch.nn.Module with a non-function custom __getattr__')
        return variables.UserMethodVariable(getattr_fn, self, **options).call_function(tx, [variables.ConstantVariable.create(name)], {})

    def var_getattr(self, tx, name):
        from .builder import VariableBuilder
        if self.source:
            source = AttrSource(self.source, name)
        else:
            source = None
        base = tx.output.get_submodule(self.module_key)
        base_dict = object.__getattribute__(base, '__dict__')
        object_member = True
        all_class_attribute_names = set()
        for x in inspect.getmro(base.__class__):
            all_class_attribute_names.update(x.__dict__.keys())
        if not self.source:
            unimplemented('GETATTR with no source')
        if name in base_dict:
            subobj = base_dict[name]
        elif '_modules' in base_dict and name in base_dict['_modules'] and (name not in all_class_attribute_names):
            subobj = base_dict['_modules'][name]
        elif '_parameters' in base_dict and name in base_dict['_parameters']:
            subobj = base_dict['_parameters'][name]
        elif '_buffers' in base_dict and name in base_dict['_buffers']:
            subobj = base_dict['_buffers'][name]
        else:
            try:
                subobj = inspect.getattr_static(base, name)
                object_member = False
            except AttributeError:
                result = self._custom_getattr_fallback(base=base, tx=tx, name=name, options={'source': source})
                if result is not None:
                    return result
                raise
        if name == '__class__' and (not object_member):
            return variables.UserDefinedClassVariable(base.__class__, source=source)
        if object_member:
            return VariableBuilder(tx, NNModuleSource(source))(subobj)
        elif istype(subobj, property):
            return variables.UserFunctionVariable(subobj.fget, source=source).call_function(tx, [self], {})
        elif istype(subobj, classmethod):
            return variables.UserMethodVariable(subobj.__func__, variables.UserDefinedObjectVariable(type(base)), source=source)
        elif istype(subobj, staticmethod):
            return variables.UserFunctionVariable(subobj.__get__(base), source=source)
        elif istype(subobj, types.FunctionType):
            return variables.UserMethodVariable(subobj, self, source=source)
        elif is_safe_constant(subobj) or istensor(subobj):
            return VariableBuilder(tx, NNModuleSource(source))(subobj)
        else:
            unimplemented(f'class property {typestr(base)} {typestr(subobj)}')
        return variables.GetAttrVariable(self, name, source=source)

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        mod = tx.output.get_submodule(self.module_key)
        with record_nn_module_stack(self.module_key, self.source, tx, mod):
            is_lazy = is_lazy_module(mod)
            if isinstance(mod, torch.nn.Sequential) and mod.__class__.forward is torch.nn.Sequential.forward:
                if nnmodule_has_hooks(mod):
                    self.convert_to_unspecialized(tx)
                assert not is_lazy, "Expected lazy sequential isn't a valid combination?"
                assert not kwargs
                arg, = args
                for child_name, submod in mod._modules.items():
                    tx.call_function(tx.output.register_attr_or_module(submod, self.module_key, child_name, source=NNModuleSource(AttrSource(self.source, child_name))), [arg], {})
                    arg = tx.pop()
                return arg
            if is_lazy:
                if mod.cls_to_become is not None:
                    self.module_type = mod.cls_to_become
                initialize_lazy_module(tx, mod, args, kwargs)
            if tx.output.is_root_tracer() and is_allowed(mod.__class__):
                if nnmodule_has_hooks(mod, check_forward_hooks=True, check_backward_hooks=True):
                    self.convert_to_unspecialized(tx)
                from .builder import wrap_fx_proxy
                return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_module', self.module_key, *proxy_args_kwargs(args, kwargs)))
            else:
                assert self.source, 'Must provide a valid source in order to inline, since inlined function may have default args which must be guarded.'
                if isinstance(mod, torch.fx.GraphModule):
                    fn = mod.forward
                else:
                    fn = mod._call_impl
                fn_source = AttrSource(self.source, '__call__')
                if istype(fn, types.MethodType):
                    fn = fn.__func__
                    fn_source = AttrSource(fn_source, '__func__')
                    args = [self] + args
                else:
                    assert istype(fn, types.FunctionType)
                return tx.inline_user_function_return(variables.UserFunctionVariable(fn, source=fn_source), args, kwargs)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]', constant=False) -> 'VariableTracker':
        from . import ConstantVariable, ListIteratorVariable, TupleVariable
        key = self.module_key
        module = tx.output.get_submodule(key)

        def generic_call_method_helper(name):
            mod_proxy = tx.output.create_proxy('get_attr', self.module_key, tuple(), {})
            mod_proxy.node.meta['example_value'] = module
            proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)
            from .builder import wrap_fx_proxy
            return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_method', name, args=(mod_proxy, *proxy_args), kwargs=proxy_kwargs))
        if name in ['_call_impl', '_wrapped_call_impl']:
            return self.call_function(tx, args, kwargs)
        elif name == 'forward':
            with record_nn_module_stack(self.module_key, self.source, tx, module):
                return generic_call_method_helper(name)
        if name == '_check_input_dim' and skipfiles.is_torch_inline_allowed(inspect.getfile(module.__class__._check_input_dim)):
            return ConstantVariable.create(True)
        if name == '_get_item_by_idx':
            assert args[1].is_python_constant()
            assert isinstance(args[0], TupleVariable)
            mod_var = args[0].items[args[1].value]
            if isinstance(mod_var, UnspecializedNNModuleVariable):
                return mod_var
            key = mod_var.module_key
            submod = tx.output.get_submodule(key)
            return tx.output.register_attr_or_module(submod, key, key, source=NNModuleSource(GetItemSource(self.source, key)))
        if constant:
            fn = getattr(module, name)
            name = f'{module.__class__.__name__}_{name}_result'
            return invoke_and_store_as_constant(tx, fn, name, args, kwargs)

        def assert_all_args_kwargs_const():
            if not all((x.is_python_constant() for x in itertools.chain(args, kwargs.values()))):
                raise unimplemented(f'non-const NNModule method {name}')

        def get_kwargs(*names):
            assert_all_args_kwargs_const()
            fn = getattr(module, name)
            bound_args = inspect.signature(fn).bind(*[x.as_python_constant() for x in args], **{k: v.as_python_constant() for k, v in kwargs.items()})
            bound_args.apply_defaults()
            bound_args = bound_args.arguments
            return {k: bound_args[k] for k in names}

        def wrap_values(items):
            result = []
            for name, submod in items:
                result.append(tx.output.register_attr_or_module(submod, key, name, source=NNModuleSource(gen_source(self.source, name))))
            return ListIteratorVariable(result, mutable_local=MutableLocal())

        def named_embed(name, obj):
            return TupleVariable([ConstantVariable.create(name), tx.output.register_attr_or_module(obj, key, name, source=NNModuleSource(gen_source(self.source, name)))])

        def gen_source(source, name):
            name_split = name.split('.')
            if name_split[0] == '':
                return source
            while len(name_split) > 0:
                x = name_split.pop(0)
                source = AttrSource(source, x)
            return source
        if name == 'named_children':
            assert not (args or kwargs)
            result = []
            for name, submod in module.named_children():
                result.append(named_embed(name, submod))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == 'named_parameters':
            result = []
            for name, param in module.named_parameters(**get_kwargs('prefix', 'recurse')):
                result.append(named_embed(name, param))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == 'named_buffers':
            result = []
            for name, buffer in module.named_buffers(**get_kwargs('prefix', 'recurse', 'remove_duplicate')):
                result.append(named_embed(name, buffer))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == 'named_modules':
            result = []
            for name, submod in module.named_modules(**get_kwargs('memo', 'prefix', 'remove_duplicate')):
                result.append(named_embed(name, submod))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == 'children':
            assert not (args or kwargs)
            return wrap_values(module.named_children())
        elif name == 'modules':
            return wrap_values(module.named_modules())
        elif name == 'parameters':
            return wrap_values(module.named_parameters(**get_kwargs('recurse')))
        elif name == 'buffers':
            return wrap_values(module.named_buffers(**get_kwargs('recurse')))
        elif name == 'keys':
            assert not (args or kwargs)
            result = []
            for name in module.keys():
                result.append(ConstantVariable.create(name))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == 'values':
            assert not (args or kwargs)
            return wrap_values(module.items())
        elif name == 'items':
            assert not (args or kwargs)
            result = []
            for name, submod in module.items():
                result.append(named_embed(name, submod))
            return ListIteratorVariable(result, mutable_local=MutableLocal())
        elif name == '__len__':
            assert not (args or kwargs)
            return ConstantVariable.create(len(module))
        elif name == '__contains__' and isinstance(module, (torch.nn.ModuleDict, torch.nn.ParameterDict)) and args and args[0].is_python_constant():
            return ConstantVariable.create(args[0].as_python_constant() in module._modules)
        elif name == '__getitem__':
            assert not kwargs and len(args) == 1
            builtin_supported = (torch.nn.ModuleDict.__getitem__, torch.nn.ModuleList.__getitem__, torch.nn.ParameterDict.__getitem__, torch.nn.ParameterList.__getitem__, torch.nn.Sequential.__getitem__)
            if type(module).__getitem__ not in builtin_supported:
                assert isinstance(args[0], variables.ConstantVariable), typestr(args[0])
                key = args[0].as_python_constant()
                assert isinstance(key, (str, int))
                fn = getattr(module, name).__func__
                assert isinstance(fn, types.FunctionType)
                src = AttrSource(AttrSource(self.source, name), '__func__')
                return tx.inline_user_function_return(variables.UserFunctionVariable(fn, source=src), [self] + list(args), kwargs)
            assert self.source
            if isinstance(args[0], SliceVariable):
                result = []
                submods = []
                keys = list(range(len(module)))[args[0].as_python_constant()]
                for idx, submod in enumerate(module[args[0].as_python_constant()]):
                    key = keys[idx]
                    src = NNModuleSource(GetItemSource(self.source, key))
                    result.append(tx.output.register_attr_or_module(submod, key, source=src))
                    submods.append(submod)
                new_module = torch.nn.Sequential(*submods)
                new_module_variable = tx.output.register_attr_or_module(new_module, f'{self}.__getitem__(slice)', source=NNModuleSource(GetItemSource(self.source, args[0].as_python_constant())))
                return new_module_variable
            key = args[0].as_python_constant()
            submod = module[key]
            return tx.output.register_attr_or_module(submod, self.module_key, key, source=NNModuleSource(GetItemSource(self.source, key)))
        elif name == '_get_abs_string_index' or (isinstance(module, torch.nn.modules.conv._ConvNd) and name == '_conv_forward') or (isinstance(module, torch.nn.modules.conv._ConvTransposeNd) and name == '_output_padding'):
            fn = getattr(module, name).__func__
            fn_source = AttrSource(AttrSource(self.source, name), '__func__')
            return tx.inline_user_function_return(variables.UserFunctionVariable(fn, source=fn_source), [self] + args, kwargs)
        elif name in module.__class__.__dict__ and callable(module.__class__.__dict__[name]) and all((isinstance(x, variables.TensorVariable) for x in itertools.chain(args, kwargs.values()))):
            return generic_call_method_helper(name)
        else:
            return super().call_method(tx, name, args, kwargs)