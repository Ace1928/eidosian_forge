from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _mock_set_magics(self):
    these_magics = _magics
    if getattr(self, '_mock_methods', None) is not None:
        these_magics = _magics.intersection(self._mock_methods)
        remove_magics = set()
        remove_magics = _magics - these_magics
        for entry in remove_magics:
            if entry in type(self).__dict__:
                delattr(self, entry)
    these_magics = these_magics - set(type(self).__dict__)
    _type = type(self)
    for entry in these_magics:
        setattr(_type, entry, MagicProxy(entry, self))