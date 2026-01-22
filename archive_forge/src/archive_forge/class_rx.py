from __future__ import annotations
import asyncio
import inspect
import math
import operator
from collections.abc import Iterable, Iterator
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Callable, Optional
from .depends import depends
from .display import _display_accessors, _reactive_display_objs
from .parameterized import (
from .parameters import Boolean, Event
from ._utils import _to_async_gen, iscoroutinefunction, full_groupby
class rx:
    """
    `rx` allows wrapping objects and then operating on them
    interactively while recording any operations applied to them. By
    recording all arguments or operands in the operations the recorded
    pipeline can be replayed if an operand represents a dynamic value.

    Parameters
    ----------
    obj: any
        A supported data structure object

    Examples
    --------
    Instantiate it from an object:

    >>> ifloat = rx(3.14)
    >>> ifloat * 2
    6.28

    Then update the original value and see the new result:
    >>> ifloat.value = 1
    2
    """
    _accessors: dict[str, Callable[[rx], Any]] = {}
    _display_options: tuple[str] = ()
    _display_handlers: dict[type, tuple[Any, dict[str, Any]]] = {}
    _method_handlers: dict[str, Callable] = {}

    @classmethod
    def register_accessor(cls, name: str, accessor: Callable[[rx], Any], predicate: Optional[Callable[[Any], bool]]=None):
        """
        Registers an accessor that extends rx with custom behavior.

        Arguments
        ---------
        name: str
          The name of the accessor will be attribute-accessible under.
        accessor: Callable[[rx], any]
          A callable that will return the accessor namespace object
          given the rx object it is registered on.
        predicate: Callable[[Any], bool] | None
        """
        cls._accessors[name] = (accessor, predicate)

    @classmethod
    def register_display_handler(cls, obj_type, handler, **kwargs):
        """
        Registers a display handler for a specific type of object,
        making it possible to define custom display options for
        specific objects.

        Arguments
        ---------
        obj_type: type | callable
          The type to register a custom display handler on.
        handler: Viewable | callable
          A Viewable or callable that is given the object to be displayed
          and the custom keyword arguments.
        kwargs: dict[str, Any]
          Additional display options to register for this type.
        """
        cls._display_handlers[obj_type] = (handler, kwargs)

    @classmethod
    def register_method_handler(cls, method, handler):
        """
        Registers a handler that is called when a specific method on
        an object is called.
        """
        cls._method_handlers[method] = handler

    def __new__(cls, obj=None, **kwargs):
        wrapper = None
        obj = transform_reference(obj)
        if kwargs.get('fn'):
            fn = kwargs.pop('fn')
            wrapper = kwargs.pop('_wrapper', None)
        elif inspect.isgeneratorfunction(obj) or iscoroutinefunction(obj):
            wrapper = GenWrapper(object=obj)
            fn = bind(lambda obj: obj, wrapper.param.object)
            obj = Undefined
        elif isinstance(obj, (FunctionType, MethodType)) and hasattr(obj, '_dinfo'):
            fn = obj
            obj = None
        elif isinstance(obj, Parameter):
            fn = bind(lambda obj: obj, obj)
            obj = getattr(obj.owner, obj.name)
        else:
            wrapper = Wrapper(object=obj)
            fn = bind(lambda obj: obj, wrapper.param.object)
        inst = super(rx, cls).__new__(cls)
        inst._fn = fn
        inst._shared_obj = kwargs.get('_shared_obj', None if obj is None else [obj])
        inst._wrapper = wrapper
        return inst

    def __init__(self, obj=None, operation=None, fn=None, depth=0, method=None, prev=None, _shared_obj=None, _current=None, _wrapper=None, **kwargs):
        self._init = False
        display_opts = {}
        for _, opts in self._display_handlers.values():
            for k, o in opts.items():
                display_opts[k] = o
        display_opts.update({dopt: kwargs.pop(dopt) for dopt in self._display_options + tuple(display_opts) if dopt in kwargs})
        self._display_opts = display_opts
        self._method = method
        self._operation = operation
        self._depth = depth
        self._dirty = _current is None
        self._dirty_obj = False
        self._current_task = None
        self._error_state = None
        self._current_ = _current
        if isinstance(obj, rx) and (not prev):
            self._prev = obj
        else:
            self._prev = prev
        if operation and (iscoroutinefunction(operation['fn']) or inspect.isgeneratorfunction(operation['fn'])):
            self._trigger = Trigger(internal=True)
            self._current_ = Undefined
        else:
            self._trigger = None
        self._root = self._compute_root()
        self._fn_params = self._compute_fn_params()
        self._internal_params = self._compute_params()
        self._params = [p for p in self._internal_params if (not isinstance(p.owner, Trigger) or p.owner.internal) or any((p not in self._internal_params for p in p.owner.parameters))]
        self._setup_invalidations(depth)
        self._kwargs = kwargs
        self.rx = reactive_ops(self)
        self._init = True
        for name, accessor in _display_accessors.items():
            setattr(self, name, accessor(self))
        for name, (accessor, predicate) in rx._accessors.items():
            if predicate is None or predicate(self._current):
                setattr(self, name, accessor(self))

    @property
    def _obj(self):
        if self._shared_obj is None:
            self._obj = eval_function_with_deps(self._fn)
        elif self._root._dirty_obj:
            root = self._root
            root._shared_obj[0] = eval_function_with_deps(root._fn)
            root._dirty_obj = False
        return self._shared_obj[0]

    @_obj.setter
    def _obj(self, obj):
        if self._shared_obj is None:
            self._shared_obj = [obj]
        else:
            self._shared_obj[0] = obj

    @property
    def _current(self):
        if self._error_state:
            raise self._error_state
        elif self._dirty or self._root._dirty_obj:
            self._resolve()
        return self._current_

    def _compute_root(self):
        if self._prev is None:
            return self
        root = self
        while root._prev is not None:
            root = root._prev
        return root

    def _compute_fn_params(self) -> list[Parameter]:
        if self._fn is None:
            return []
        owner = get_method_owner(self._fn)
        if owner is not None:
            deps = [dep.pobj for dep in owner.param.method_dependencies(self._fn.__name__)]
            return deps
        dinfo = getattr(self._fn, '_dinfo', {})
        args = list(dinfo.get('dependencies', []))
        kwargs = list(dinfo.get('kw', {}).values())
        return args + kwargs

    def _compute_params(self) -> list[Parameter]:
        ps = list(self._fn_params)
        if self._trigger:
            ps.append(self._trigger.param.value)
        prev = self._prev
        while prev is not None:
            for p in prev._params:
                if p not in ps:
                    ps.append(p)
            prev = prev._prev
        if self._operation is None:
            return ps
        for ref in resolve_ref(self._operation['fn']):
            if ref not in ps:
                ps.append(ref)
        for arg in list(self._operation['args']) + list(self._operation['kwargs'].values()):
            for ref in resolve_ref(arg):
                if ref not in ps:
                    ps.append(ref)
        return ps

    def _setup_invalidations(self, depth: int=0):
        """
        Since the parameters of the pipeline can change at any time
        we have to invalidate the internal state of the pipeline.
        To handle both invalidations of the inputs of the pipeline
        and the pipeline itself we set up watchers on both.

        1. The first invalidation we have to set up is to re-evaluate
           the function that feeds the pipeline. Only the root node of
           a pipeline has to perform this invalidation because all
           leaf nodes inherit the same shared_obj. This avoids
           evaluating the same function for every branch of the pipeline.
        2. The second invalidation is for the pipeline itself, i.e.
           if any parameter changes we have to notify the pipeline that
           it has to re-evaluate the pipeline. This is done by marking
           the pipeline as `_dirty`. The next time the `_current` value
           is requested the value is resolved by re-executing the
           pipeline.
        """
        if self._fn is not None:
            for _, params in full_groupby(self._fn_params, lambda x: id(x.owner)):
                fps = [p.name for p in params if p in self._root._fn_params]
                if fps:
                    params[0].owner.param._watch(self._invalidate_obj, fps, precedence=-1)
        for _, params in full_groupby(self._internal_params, lambda x: id(x.owner)):
            params[0].owner.param._watch(self._invalidate_current, [p.name for p in params], precedence=-1)

    def _invalidate_current(self, *events):
        if all((event.obj is self._trigger for event in events)):
            return
        self._dirty = True
        self._error_state = None

    def _invalidate_obj(self, *events):
        self._root._dirty_obj = True
        self._error_state = None

    async def _resolve_async(self, obj):
        self._current_task = task = asyncio.current_task()
        if inspect.isasyncgen(obj):
            async for val in obj:
                if self._current_task is not task:
                    break
                self._current_ = val
                self._trigger.param.trigger('value')
        else:
            value = await obj
            if self._current_task is task:
                self._current_ = value
                self._trigger.param.trigger('value')

    def _lazy_resolve(self, obj):
        from .parameterized import async_executor
        if inspect.isgenerator(obj):
            obj = _to_async_gen(obj)
        async_executor(partial(self._resolve_async, obj))

    def _resolve(self):
        if self._error_state:
            raise self._error_state
        elif self._dirty or self._root._dirty_obj:
            try:
                obj = self._obj if self._prev is None else self._prev._resolve()
                if obj is Skip or obj is Undefined:
                    self._current_ = Undefined
                    raise Skip
                operation = self._operation
                if operation:
                    obj = self._eval_operation(obj, operation)
                    if inspect.isasyncgen(obj) or inspect.iscoroutine(obj) or inspect.isgenerator(obj):
                        self._lazy_resolve(obj)
                        obj = Skip
                    if obj is Skip:
                        raise Skip
            except Skip:
                self._dirty = False
                return self._current_
            except Exception as e:
                self._error_state = e
                raise e
            self._current_ = current = obj
        else:
            current = self._current_
        self._dirty = False
        if self._method:
            current = getattr(current, self._method, current)
        if hasattr(current, '__call__'):
            self.__call__.__func__.__doc__ = self.__call__.__doc__
        return current

    def _transform_output(self, obj):
        """
        Applies custom display handlers before their output.
        """
        applies = False
        for predicate, (handler, opts) in self._display_handlers.items():
            display_opts = {k: v for k, v in self._display_opts.items() if k in opts}
            display_opts.update(self._kwargs)
            try:
                applies = predicate(obj, **display_opts)
            except TypeError:
                applies = predicate(obj)
            if applies:
                new = handler(obj, **display_opts)
                if new is not obj:
                    return new
        return obj

    @property
    def _callback(self):
        params = self._params

        def evaluate(*args, **kwargs):
            return self._transform_output(self._current)
        if params:
            return bind(evaluate, *params)
        return evaluate

    def _clone(self, operation=None, copy=False, **kwargs):
        operation = operation or self._operation
        depth = self._depth + 1
        if copy:
            kwargs = dict(self._kwargs, _current=self._current, method=self._method, prev=self._prev, **kwargs)
        else:
            kwargs = dict(prev=self, **dict(self._kwargs, **kwargs))
        kwargs = dict(self._display_opts, **kwargs)
        return type(self)(self._obj, operation=operation, depth=depth, fn=self._fn, _shared_obj=self._shared_obj, _wrapper=self._wrapper, **kwargs)

    def __dir__(self):
        current = self._current
        if self._method:
            current = getattr(current, self._method)
        extras = {attr for attr in dir(current) if not attr.startswith('_')}
        try:
            return sorted(set(super().__dir__()) | extras)
        except Exception:
            return sorted(set(dir(type(self))) | set(self.__dict__) | extras)

    def _resolve_accessor(self):
        if not self._method:
            return self._clone(copy=True)
        operation = {'fn': getattr, 'args': (self._method,), 'kwargs': {}, 'reverse': False}
        self._method = None
        return self._clone(operation)

    def __getattribute__(self, name):
        self_dict = super().__getattribute__('__dict__')
        if not self_dict.get('_init') or name == 'rx' or name.startswith('_'):
            return super().__getattribute__(name)
        current = self_dict['_current_']
        dirty = self_dict['_dirty']
        if dirty:
            self._resolve()
            current = self_dict['_current_']
        method = self_dict['_method']
        if method:
            current = getattr(current, method)
        extras = [d for d in dir(current) if not d.startswith('_')]
        if (name in extras or current is Undefined) and name not in super().__dir__():
            new = self._resolve_accessor()
            new._method = name
            try:
                new.__doc__ = getattr(current, name).__doc__
            except Exception:
                pass
            return new
        return super().__getattribute__(name)

    def __call__(self, *args, **kwargs):
        new = self._clone(copy=True)
        method = new._method or '__call__'
        if method == '__call__' and self._depth == 0 and (not hasattr(self._current, '__call__')):
            return self.set_display(*args, **kwargs)
        if method in rx._method_handlers:
            handler = rx._method_handlers[method]
            method = handler(self)
        new._method = None
        kwargs = dict(kwargs)
        operation = {'fn': method, 'args': args, 'kwargs': kwargs, 'reverse': False}
        return new._clone(operation)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        new = self._resolve_accessor()
        operation = {'fn': getattr(ufunc, method), 'args': args[1:], 'kwargs': kwargs, 'reverse': False}
        return new._clone(operation)

    def _apply_operator(self, operator, *args, reverse=False, **kwargs):
        new = self._resolve_accessor()
        operation = {'fn': operator, 'args': args, 'kwargs': kwargs, 'reverse': reverse}
        return new._clone(operation)

    def __abs__(self):
        return self._apply_operator(abs)

    def __str__(self):
        return self._apply_operator(str)

    def __round__(self, ndigits=None):
        args = () if ndigits is None else (ndigits,)
        return self._apply_operator(round, *args)

    def __ceil__(self):
        return self._apply_operator(math.ceil)

    def __floor__(self):
        return self._apply_operator(math.floor)

    def __invert__(self):
        return self._apply_operator(operator.inv)

    def __neg__(self):
        return self._apply_operator(operator.neg)

    def __pos__(self):
        return self._apply_operator(operator.pos)

    def __trunc__(self):
        return self._apply_operator(math.trunc)

    def __add__(self, other):
        return self._apply_operator(operator.add, other)

    def __and__(self, other):
        return self._apply_operator(operator.and_, other)

    def __contains_(self, other):
        return self._apply_operator(operator.contains, other)

    def __divmod__(self, other):
        return self._apply_operator(divmod, other)

    def __eq__(self, other):
        return self._apply_operator(operator.eq, other)

    def __floordiv__(self, other):
        return self._apply_operator(operator.floordiv, other)

    def __ge__(self, other):
        return self._apply_operator(operator.ge, other)

    def __gt__(self, other):
        return self._apply_operator(operator.gt, other)

    def __le__(self, other):
        return self._apply_operator(operator.le, other)

    def __lt__(self, other):
        return self._apply_operator(operator.lt, other)

    def __lshift__(self, other):
        return self._apply_operator(operator.lshift, other)

    def __matmul__(self, other):
        return self._apply_operator(operator.matmul, other)

    def __mod__(self, other):
        return self._apply_operator(operator.mod, other)

    def __mul__(self, other):
        return self._apply_operator(operator.mul, other)

    def __ne__(self, other):
        return self._apply_operator(operator.ne, other)

    def __or__(self, other):
        return self._apply_operator(operator.or_, other)

    def __rshift__(self, other):
        return self._apply_operator(operator.rshift, other)

    def __pow__(self, other):
        return self._apply_operator(operator.pow, other)

    def __sub__(self, other):
        return self._apply_operator(operator.sub, other)

    def __truediv__(self, other):
        return self._apply_operator(operator.truediv, other)

    def __xor__(self, other):
        return self._apply_operator(operator.xor, other)

    def __radd__(self, other):
        return self._apply_operator(operator.add, other, reverse=True)

    def __rand__(self, other):
        return self._apply_operator(operator.and_, other, reverse=True)

    def __rdiv__(self, other):
        return self._apply_operator(operator.div, other, reverse=True)

    def __rdivmod__(self, other):
        return self._apply_operator(divmod, other, reverse=True)

    def __rfloordiv__(self, other):
        return self._apply_operator(operator.floordiv, other, reverse=True)

    def __rlshift__(self, other):
        return self._apply_operator(operator.rlshift, other)

    def __rmod__(self, other):
        return self._apply_operator(operator.mod, other, reverse=True)

    def __rmul__(self, other):
        return self._apply_operator(operator.mul, other, reverse=True)

    def __ror__(self, other):
        return self._apply_operator(operator.or_, other, reverse=True)

    def __rpow__(self, other):
        return self._apply_operator(operator.pow, other, reverse=True)

    def __rrshift__(self, other):
        return self._apply_operator(operator.rrshift, other)

    def __rsub__(self, other):
        return self._apply_operator(operator.sub, other, reverse=True)

    def __rtruediv__(self, other):
        return self._apply_operator(operator.truediv, other, reverse=True)

    def __rxor__(self, other):
        return self._apply_operator(operator.xor, other, reverse=True)

    def __getitem__(self, other):
        return self._apply_operator(operator.getitem, other)

    def __iter__(self):
        if isinstance(self._current, Iterator):
            while True:
                try:
                    new = self._apply_operator(next)
                    new.rx.value
                except RuntimeError:
                    break
                yield new
            return
        elif not isinstance(self._current, Iterable):
            raise TypeError(f'cannot unpack non-iterable {type(self._current).__name__} object.')
        items = self._apply_operator(list)
        for i in range(len(self._current)):
            yield items[i]

    def _eval_operation(self, obj, operation):
        fn, args, kwargs = (operation['fn'], operation['args'], operation['kwargs'])
        resolved_args = []
        for arg in args:
            val = resolve_value(arg)
            if val is Skip or val is Undefined:
                raise Skip
            resolved_args.append(val)
        resolved_kwargs = {}
        for k, arg in kwargs.items():
            val = resolve_value(arg)
            if val is Skip or val is Undefined:
                raise Skip
            resolved_kwargs[k] = val
        if isinstance(fn, str):
            obj = getattr(obj, fn)(*resolved_args, **resolved_kwargs)
        elif operation.get('reverse'):
            obj = fn(resolved_args[0], obj, *resolved_args[1:], **resolved_kwargs)
        else:
            obj = fn(obj, *resolved_args, **resolved_kwargs)
        return obj