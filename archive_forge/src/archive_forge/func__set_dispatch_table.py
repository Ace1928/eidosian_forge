import copyreg
import io
import functools
import types
import sys
import os
from multiprocessing import util
from pickle import loads, HIGHEST_PROTOCOL
def _set_dispatch_table(self, dispatch_table):
    for ancestor_class in self._loky_pickler_cls.mro():
        dt_attribute = getattr(ancestor_class, 'dispatch_table', None)
        if isinstance(dt_attribute, types.MemberDescriptorType):
            dt_attribute.__set__(self, dispatch_table)
            break
    self.dispatch_table = dispatch_table