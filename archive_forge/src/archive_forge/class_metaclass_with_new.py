import dill
from enum import EnumMeta
import sys
from collections import namedtuple
class metaclass_with_new(type):

    def __new__(mcls, name, bases, ns, **kwds):
        cls = super().__new__(mcls, name, bases, ns, **kwds)
        assert mcls is not None
        assert cls.method(mcls)
        return cls

    def method(cls, mcls):
        return isinstance(cls, mcls)