from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadReference(_messages.Message):
    """Reference of an underlying compute resource represented by the Workload.

  Fields:
    uri: Output only. The underlying compute resource uri.
  """
    uri = _messages.StringField(1)