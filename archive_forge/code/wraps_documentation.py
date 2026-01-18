from __future__ import annotations
from typing import Type, TypeVar, Optional, Union, Any, Callable, List, Dict, overload, TYPE_CHECKING
from .main import ProxyObject

    Lazily initialize an object as a proxy object

    Args:
    - obj_cls: The object class
    - obj_getter: The object getter. This can be a callable or a string which will lazily import the object
    - obj_args: The object arguments. This can be a list or a callable that returns a list, or a string that lazily imports a function/list
    - obj_kwargs: The object keyword arguments. This can be a dictionary or a callable that returns a dictionary, or a string that lazily imports a function/dictionary
    - obj_initialize: The object initialization flag
    - threadsafe: The thread safety flag

    Returns:
    - Type[OT]: The proxy object

    Usage:
    @proxied
    class DummyClass(abc.ABC):

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            print('DummyClass initialized')
    
    x = DummyClass
    >> DummyClass initialized

    # Note that if you initialize the object directly, it will raise an error
    y = DummyClass() # Raises an error

    def dummy_args_getter() -> Iterable:
        return [1, 2]

    # @proxied(obj_args=(1, 2), obj_kwargs={'a': 1, 'b': 2})
    @proxied(obj_args=dummy_args_getter, obj_kwargs={'a': 1, 'b': 2})
    class DummyClass(abc.ABC):

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            print('DummyClass initialized', args, kwargs)
    
    x = DummyClass
    >> DummyClass initialized (1, 2) {'a': 1, 'b': 2}
    