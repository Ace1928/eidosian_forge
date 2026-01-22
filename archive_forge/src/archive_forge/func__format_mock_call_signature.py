from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _format_mock_call_signature(self, args, kwargs):
    name = self._mock_name or 'mock'
    return _format_call_signature(name, args, kwargs)