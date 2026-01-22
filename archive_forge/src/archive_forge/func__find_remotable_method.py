from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
def _find_remotable_method(self, cls, thing, parent_was_remotable=False):
    """Follow a chain of remotable things down to the original function."""
    if isinstance(thing, classmethod):
        return self._find_remotable_method(cls, thing.__get__(None, cls))
    elif (inspect.ismethod(thing) or inspect.isfunction(thing)) and hasattr(thing, 'remotable'):
        return self._find_remotable_method(cls, thing.original_fn, parent_was_remotable=True)
    elif parent_was_remotable:
        return thing
    else:
        return None