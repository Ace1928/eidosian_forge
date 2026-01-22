from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunTransferJobRequest(_messages.Message):
    """Request passed to RunTransferJob.

  Fields:
    projectId: Required. The ID of the Google Cloud project that owns the
      transfer job.
  """
    projectId = _messages.StringField(1)