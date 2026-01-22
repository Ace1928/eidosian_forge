from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Guaranteed(_messages.Message):
    """Guaranteed tier definition.

  Fields:
    minDuration: Optional. Defines the minimum duration of the guarantee. If
      specified, the requested resources will only be provisioned if they can
      be allocated for at least the given duration.
  """
    minDuration = _messages.StringField(1)