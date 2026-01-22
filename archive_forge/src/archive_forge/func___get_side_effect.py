from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def __get_side_effect(self):
    delegated = self._mock_delegate
    if delegated is None:
        return self._mock_side_effect
    sf = delegated.side_effect
    if sf is not None and (not callable(sf)) and (not isinstance(sf, _MockIter)) and (not _is_exception(sf)):
        sf = _MockIter(sf)
        delegated.side_effect = sf
    return sf