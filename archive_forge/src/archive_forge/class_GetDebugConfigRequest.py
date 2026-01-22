from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetDebugConfigRequest(_messages.Message):
    """Request to get updated debug configuration for component.

  Fields:
    componentId: The internal component id for which debug configuration is
      requested.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the job specified by job_id.
    workerId: The worker id, i.e., VM hostname.
  """
    componentId = _messages.StringField(1)
    location = _messages.StringField(2)
    workerId = _messages.StringField(3)