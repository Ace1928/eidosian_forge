from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutorResourceRequest(_messages.Message):
    """Resources used per executor used by the application.

  Fields:
    amount: A string attribute.
    discoveryScript: A string attribute.
    resourceName: A string attribute.
    vendor: A string attribute.
  """
    amount = _messages.IntegerField(1)
    discoveryScript = _messages.StringField(2)
    resourceName = _messages.StringField(3)
    vendor = _messages.StringField(4)