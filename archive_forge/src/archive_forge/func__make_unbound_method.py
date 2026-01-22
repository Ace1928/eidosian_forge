from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def _make_unbound_method(self):

    def _method(cls_or_self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(cls_or_self, *self.args, *args, **keywords)
    _method.__isabstractmethod__ = self.__isabstractmethod__
    _method._partialmethod = self
    return _method