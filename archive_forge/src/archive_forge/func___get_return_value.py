from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def __get_return_value(self):
    ret = self._mock_return_value
    if self._mock_delegate is not None:
        ret = self._mock_delegate.return_value
    if ret is DEFAULT:
        ret = self._get_child_mock(_new_parent=self, _new_name='()')
        self.return_value = ret
    return ret