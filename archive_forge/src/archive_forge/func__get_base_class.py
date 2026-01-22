import sys
import dis
from typing import List, Tuple, TypeVar
from types import FunctionType
def _get_base_class(components, namespace):
    try:
        obj = namespace[components[0]]
    except KeyError:
        if isinstance(namespace['__builtins__'], dict):
            obj = namespace['__builtins__'][components[0]]
        else:
            obj = getattr(namespace['__builtins__'], components[0])
    for component in components[1:]:
        if hasattr(obj, component):
            obj = getattr(obj, component)
    return obj