import builtins
import copy
import functools
import inspect
import math
import os
import warnings
import collections
from itertools import chain
from types import CodeType, FunctionType, ModuleType
from typing import (
import torch
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from ._compatibility import compatibility
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .graph_module import GraphModule
from .node import Argument, base_types, map_aggregate
from .proxy import ParameterProxy, Proxy, TracerBase, Scope, ScopeContextManager
@compatibility(is_backward_compatible=True)
class Tracer(TracerBase):
    """Tracer(autowrap_modules=(math,), autowrap_functions=())

    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.

    Tracer can be subclassed to override various behaviors of the tracing
    process. The different behaviors that can be overridden are described
    in the docstrings of the methods on this class.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, autowrap_modules: Tuple[ModuleType]=(math,), autowrap_functions: Tuple[Callable, ...]=(), param_shapes_constant: bool=False) -> None:
        """
        Construct a Tracer object.

        Args:

            autowrap_modules (Tuple[ModuleType]): defaults to `(math, )`,
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap(). Backward-compatibility for
                this parameter is guaranteed.

            autowrap_functions (Tuple[Callable, ...]): defaults to `()`,
                Python functions that should be wrapped automatically without
                needing to use fx.wrap(). Backward compatibility for this
                parameter is guaranteed.

            param_shapes_constant (bool): When this flag is set,  calls to shape,
                size and a few other shape like attributes of a module's parameter
                will be evaluated directly, rather than returning a new Proxy value
                for an attribute access. Backward compatibility for this parameter
                is guaranteed.
        """
        super().__init__()
        self._autowrap_function_ids: Set[int] = {id(value) for name, value in chain(*[m.__dict__.items() for m in autowrap_modules]) if not name.startswith('_') and callable(value)}
        self._autowrap_function_ids.update({id(f) for f in autowrap_functions})
        self._autowrap_search: List[ModuleType] = list(autowrap_modules)
        self.param_shapes_constant = param_shapes_constant
        self.submodule_paths: Optional[Dict[torch.nn.Module, str]] = None
        self.root_module_name: str = ''
        self.scope = Scope('', None)
        self.module_stack = collections.OrderedDict()
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> 'Argument':
        """
        A method to specify the behavior of tracing when preparing values to
        be used as arguments to nodes in the ``Graph``.

        By default, the behavior includes:

        #. Iterate through collection types (e.g. tuple, list, dict) and recursively
           call ``create_args`` on the elements.
        #. Given a Proxy object, return a reference to the underlying IR ``Node``
        #. Given a non-Proxy Tensor object, emit IR for various cases:

            * For a Parameter, emit a ``get_attr`` node referring to that Parameter
            * For a non-Parameter Tensor, store the Tensor away in a special
              attribute referring to that attribute.

        This method can be overridden to support more types.

        Args:

            a (Any): The value to be emitted as an ``Argument`` in the ``Graph``.


        Returns:

            The value ``a`` converted into the appropriate ``Argument``
        """
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            raise NameError('parameter is not a member of this module')
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {})
        elif isinstance(a, torch.nn.Module):
            for n_, p_ in self.root.named_modules():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {})
        if isinstance(a, tuple) and hasattr(a, '_fields'):
            args = tuple((self.create_arg(elem) for elem in a))
            return self.create_node('call_function', a.__class__, args, {})
        if isinstance(a, (torch.Tensor, ScriptObject)):
            qualname: Optional[str] = self.tensor_attrs.get(a)
            if not qualname:
                i = 0
                while True:
                    qualname = f'_tensor_constant{i}'
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                self.tensor_attrs[a] = qualname
                setattr(self.root, qualname, a)
            return self.create_node('get_attr', qualname, (), {})
        if type(a) in _proxyable_classes:
            i = 0
            while True:
                qualname = f'_{a.__class__.__name__}_constant_{i}'
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)
            return self.create_node('get_attr', qualname, (), {})
        return super().create_arg(a)

    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:

            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        return (m.__module__.startswith('torch.nn') or m.__module__.startswith('torch.ao.nn')) and (not isinstance(m, torch.nn.Sequential))

    @compatibility(is_backward_compatible=True)
    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        Helper method to find the qualified name of ``mod`` in the Module hierarchy
        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has
        a submodule named ``bar``, passing ``bar`` into this function will return
        the string "foo.bar".

        Args:

            mod (str): The ``Module`` to retrieve the qualified name for.
        """
        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError('module is not installed as a submodule')
            assert isinstance(path, str)
            return path
        else:
            for n, p in self.root.named_modules():
                if mod is p:
                    return n
            raise NameError('module is not installed as a submodule')

    @compatibility(is_backward_compatible=True)
    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.

        Args:

            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Return:

            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
        module_qualified_name = self.path_of_module(m)
        with ScopeContextManager(self.scope, Scope(module_qualified_name, type(m))) as _scope:
            self.module_stack[_scope.module_path] = (module_qualified_name, _scope.module_type)
            if not self.is_leaf_module(m, module_qualified_name):
                ret_val = forward(*args, **kwargs)
            else:
                ret_val = self.create_proxy('call_module', module_qualified_name, args, kwargs)
            key, _ = self.module_stack.popitem(last=True)
            assert key == _scope.module_path, f' Unexpected key {key}'
        return ret_val

    @compatibility(is_backward_compatible=False)
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        """
        Method that specifies the behavior of this ``Tracer`` when we call getattr
        on a call to an ``nn.Module`` instance.

        By default, the behavior is to return a proxy value for the attribute. It
        also stores the proxy value in the ``parameter_proxy_cache``, so that future
        calls will reuse the proxy rather than creating a new one.

        This method can be overridden to --for example-- not return proxies when
        querying parameters.

        Args:

            attr (str): The name of the attribute being queried
            attr_val (Any): The value of the attribute
            parameter_proxy_cache (Dict[str, Any]): A cache of attr names to proxies

        Return:

            The return value from the getattr call.
        """

        def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if 'proxy_factory_fn' in inspect.signature(self.create_proxy).parameters:
                            kwargs['proxy_factory_fn'] = None if not self.param_shapes_constant else lambda node: ParameterProxy(self, node, n, attr_val)
                        val_proxy = self.create_proxy('get_attr', n, (), {}, **kwargs)
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None
        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_parameters(), parameter_proxy_cache)
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy
        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_buffers(), parameter_proxy_cache)
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy
        return attr_val

    @compatibility(is_backward_compatible=False)
    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        """
        Create ``placeholder`` nodes corresponding to the signature of the ``root``
        Module. This method introspects root's signature and emits those
        nodes accordingly, also supporting ``*args`` and ``**kwargs``.
        """
        fn_for_analysis = inspect.unwrap(root_fn)
        co = fn_for_analysis.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        orig_args = list(co.co_varnames)
        names_iter = iter(co.co_varnames)
        args: List[Any] = []
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError('``self`` argument cannot be part of *args expansion!')
            skip_arg_idx = 1
            next(names_iter)
            args.append(self.root)
        sig = inspect.signature(fn_for_analysis)

        def proxy_placeholder(name: str):
            if concrete_args is not None and name in concrete_args:
                cnt = 0

                def replace_ph(x):
                    nonlocal cnt
                    cnt += 1
                    param = sig.parameters[name]
                    default = () if param.default is inspect.Parameter.empty else (param.default,)
                    out = self.create_proxy('placeholder', f'{name}_{str(cnt)}', default, {})
                    if isinstance(x, PHBase):

                        def transfer_attrs(fr, to):
                            for attr_name in dir(fr):
                                attr_val = getattr(fr, attr_name)
                                if not callable(attr_val) and (not attr_name.startswith('__')) and (not hasattr(to, attr_name)):
                                    setattr(to, attr_name, attr_val)
                        if x != PH:
                            transfer_attrs(fr=x, to=out.node)
                        return out
                    if type(x) == bool or (type(x) in base_types and type(x) != torch.Tensor):
                        torch._assert(out == x, f'{name} has been specialized to have value {x} but got another value')
                    elif type(x) == type(None):
                        args = (out, f'{name} has been specialized to have value None but got another value')
                        self.create_proxy('call_function', _assert_is_none, args, {})
                    else:
                        warnings.warn(f'Was not able to add assertion to guarantee correct input {name} to specialized function. It is up to the user to make sure that your inputs match the inputs you specialized the function with.')
                    return x
                return pytree.tree_map(replace_ph, concrete_args[name])
            if name[0] == '*':
                default = ()
            else:
                param = sig.parameters[name]
                default = () if param.default is inspect.Parameter.empty else (param.default,)
            return self.create_proxy('placeholder', name, default, {}, type_expr=fn_for_analysis.__annotations__.get(name, None))
        arg_names = [next(names_iter) for idx in range(skip_arg_idx, total_args)]
        if isinstance(concrete_args, tuple):
            if len(arg_names) != len(concrete_args):
                raise RuntimeError(f'Tracing expected {len(arg_names)} arguments but got {len(concrete_args)} concrete arguments')
            concrete_args = dict(zip(arg_names, concrete_args))
        args.extend((proxy_placeholder(names) for names in arg_names))
        if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
            if co.co_flags & inspect.CO_VARARGS:
                args.append(proxy_placeholder('*' + next(names_iter)))
            if co.co_flags & inspect.CO_VARKEYWORDS:
                args.append(proxy_placeholder('**' + next(names_iter)))
            root_fn = _patch_function(root_fn, len(args))
        flat_args, in_spec = pytree.tree_flatten(tuple(args))
        if any((not isinstance(i, pytree.LeafSpec) for i in in_spec.children_specs)):
            self.graph._codegen = _PyTreeCodeGen(_PyTreeInfo(orig_args[:total_args], in_spec, None))

            def flatten_fn(*args):
                tree_args = pytree.tree_unflatten(list(args), in_spec)
                tree_out = root_fn(*tree_args)
                out_args, out_spec = pytree.tree_flatten(tree_out)
                assert isinstance(self.graph._codegen, _PyTreeCodeGen)
                self.graph._codegen.pytree_info = self.graph._codegen.pytree_info._replace(out_spec=out_spec)
                return out_args
            return (flatten_fn, flat_args)
        return (root_fn, args)

    @compatibility(is_backward_compatible=True)
    def trace(self, root: Union[torch.nn.Module, Callable[..., Any]], concrete_args: Optional[Dict[str, Any]]=None) -> Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants
        to.


        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
                traced through. Backwards-compatibility for this parameter is
                guaranteed.
            concrete_args (Optional[Dict[str, any]]): Concrete arguments that should
                not be treated as Proxies. This parameter is experimental and
                its backwards-compatibility is *NOT* guaranteed.

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        global _is_fx_tracing_flag
        old_is_fx_tracing_flag = _is_fx_tracing_flag
        _is_fx_tracing_flag = True
        try:
            if isinstance(root, torch.nn.Module):
                self.root = root
                assert hasattr(type(root), self.traced_func_name), f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"
                fn = getattr(type(root), self.traced_func_name)
                self.root_module_name = root._get_name()
                self.submodule_paths = {mod: name for name, mod in root.named_modules()}
            else:
                self.root = torch.nn.Module()
                fn = root
            tracer_cls: Optional[Type[Tracer]] = getattr(self, '__class__', None)
            self.graph = Graph(tracer_cls=tracer_cls)
            if hasattr(fn, '__code__'):
                code = fn.__code__
                self.graph._co_fields = {'co_name': code.co_name, 'co_filename': code.co_filename, 'co_firstlineno': code.co_firstlineno}
            self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

            def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
                for k, v in m.__dict__.items():
                    if isinstance(v, (torch.Tensor, ScriptObject)):
                        self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
                for k, v in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])
            collect_tensor_attrs(self.root, [])
            assert isinstance(fn, FunctionType)
            fn_globals = fn.__globals__
            fn, args = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)
            parameter_proxy_cache: Dict[str, Proxy] = {}

            @functools.wraps(_orig_module_getattr)
            def module_getattr_wrapper(mod, attr):
                attr_val = _orig_module_getattr(mod, attr)
                return self.getattr(attr, attr_val, parameter_proxy_cache)

            @functools.wraps(_orig_module_call)
            def module_call_wrapper(mod, *args, **kwargs):

                def forward(*args, **kwargs):
                    return _orig_module_call(mod, *args, **kwargs)
                _autowrap_check(patcher, getattr(getattr(mod, 'forward', mod), '__globals__', {}), self._autowrap_function_ids)
                return self.call_module(mod, forward, args, kwargs)
            with _Patcher() as patcher:
                patcher.patch_method(torch.nn.Module, '__getattr__', module_getattr_wrapper, deduplicate=False)
                patcher.patch_method(torch.nn.Module, '__call__', module_call_wrapper, deduplicate=False)
                _patch_wrapped_functions(patcher)
                _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                for module in self._autowrap_search:
                    _autowrap_check(patcher, module.__dict__, self._autowrap_function_ids)
                self.create_node('output', 'output', (self.create_arg(fn(*args)),), {}, type_expr=fn.__annotations__.get('return', None))
            self.submodule_paths = None
        finally:
            _is_fx_tracing_flag = old_is_fx_tracing_flag
        return self.graph

    def __deepcopy__(self, memo):
        new_tracer = Tracer.__new__(Tracer)
        for k, v in self.__dict__.items():
            if k in {'_autowrap_search'}:
                new_obj = copy.copy(v)
            else:
                new_obj = copy.deepcopy(v, memo)
            new_tracer.__dict__[k] = new_obj
        return new_tracer