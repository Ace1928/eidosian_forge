from enum import Enum
from functools import lru_cache
from typing import (
Converts special typing forms (Union[...], Optional[...]), and parametrized
    generics (List[...], Dict[...]) into a 2-tuple of its base type and arguments.
    If ``type_`` is a regular type, or an object, this function will return
    ``None``.

    Note that this function will only perform one level of recursion - the
    arguments of nested types will not be resolved:

        >>> resolve_special_type(List[List[int]])
        (<class 'list'>, [<class 'list'>])

    Further examples:
        >>> resolve_special_type(Union[str, int])
        (typing.Union, [<class 'str'>, <class 'int'>])
        >>> resolve_special_type(List[int])
        (<class 'list'>, [<class 'int'>])
        >>> resolve_special_type(List)
        (<class 'list'>, [])
        >>> resolve_special_type(list)
        None
    