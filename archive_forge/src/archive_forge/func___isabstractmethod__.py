from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
@property
def __isabstractmethod__(self):
    return getattr(self.func, '__isabstractmethod__', False)