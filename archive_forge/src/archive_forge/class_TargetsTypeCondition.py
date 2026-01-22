from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetsTypeCondition(_messages.Message):
    """TargetsTypeCondition contains information on whether the Targets defined
  in the Delivery Pipeline are of the same type.

  Fields:
    errorDetails: Human readable error message.
    status: True if the targets are all a comparable type. For example this is
      true if all targets are GKE clusters. This is false if some targets are
      Cloud Run targets and others are GKE clusters.
  """
    errorDetails = _messages.StringField(1)
    status = _messages.BooleanField(2)