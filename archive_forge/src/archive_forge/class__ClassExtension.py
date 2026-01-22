from typing import Callable, Dict, Optional, Type, get_type_hints, Any
import inspect
from triad.utils.assertion import assert_or_throw
class _ClassExtension:

    def __init__(self, class_type: Type):
        self._class_type = class_type
        self._built_in = set(dir(class_type))
        self._ext: Dict[str, Callable] = {}

    def add_method(self, func: Callable, name: Optional[str]=None, on_dup: str='error') -> None:
        assert_or_throw(name not in self._built_in, ValueError(f'{name} is a built in attribute'))
        if name is None:
            name = func.__name__
        if name in self._ext:
            if on_dup == 'ignore':
                return
            if on_dup == 'error':
                raise ValueError(f'{name} is already registered')
        self._ext[name] = func
        setattr(self._class_type, name, func)