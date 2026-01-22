from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
class _SentinelObject(object):
    """A unique, named, sentinel object."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'sentinel.%s' % self.name