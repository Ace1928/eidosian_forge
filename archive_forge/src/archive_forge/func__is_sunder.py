import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def _is_sunder(name):
    """
    Returns True if a _sunder_ name, False otherwise.
    """
    return len(name) > 2 and name[0] == name[-1] == '_' and (name[1:2] != '_') and (name[-2:-1] != '_')