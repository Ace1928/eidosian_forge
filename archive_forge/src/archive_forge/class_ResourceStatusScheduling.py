from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceStatusScheduling(_messages.Message):
    """A ResourceStatusScheduling object.

  Fields:
    terminationTimestamp: Time in future when the instance will be terminated
      in RFC3339 text format.
  """
    terminationTimestamp = _messages.StringField(1)