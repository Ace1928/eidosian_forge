from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceUsedBy(_messages.Message):
    """A BackendServiceUsedBy object.

  Fields:
    reference: A string attribute.
  """
    reference = _messages.StringField(1)