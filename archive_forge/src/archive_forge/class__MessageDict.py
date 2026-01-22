from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
class _MessageDict(dict):
    """An extensible wrapper around message data.

  Fields can be added as dictionary items and retrieved as attributes.
  """

    def __init__(self, message, hidden_fields=None):
        super(_MessageDict, self).__init__()
        self._orig_type = type(message).__name__
        if hidden_fields:
            self._hidden_fields = hidden_fields
        else:
            self._hidden_fields = {}
        for field in message.all_fields():
            value = getattr(message, field.name)
            if not value:
                self._hidden_fields[field.name] = value
            else:
                self[field.name] = value

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        if attr in self._hidden_fields:
            return self._hidden_fields[attr]
        raise AttributeError('Type "{0}" does not have attribute "{1}"'.format(self._orig_type, attr))

    def HideExistingField(self, field_name):
        if field_name in self._hidden_fields:
            return
        self._hidden_fields[field_name] = self.pop(field_name, None)