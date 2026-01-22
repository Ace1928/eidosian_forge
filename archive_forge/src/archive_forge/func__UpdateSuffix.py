from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
def _UpdateSuffix(self, suffix):
    """Updates the suffix for this message."""
    if not isinstance(suffix, six.string_types):
        raise TypeError('expected a string or other character buffer object')
    self._suffix = suffix