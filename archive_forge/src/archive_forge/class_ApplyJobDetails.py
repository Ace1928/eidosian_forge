from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyJobDetails(_messages.Message):
    """Details regarding an Apply background job.

  Fields:
    connectionProfile: Output only. The connection profile which was used for
      the apply job.
    filter: Output only. AIP-160 based filter used to specify the entities to
      apply
  """
    connectionProfile = _messages.StringField(1)
    filter = _messages.StringField(2)