from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryJobsGetRequest(_messages.Message):
    """A BigqueryJobsGetRequest object.

  Fields:
    jobId: [Required] Job ID of the requested job
    projectId: [Required] Project ID of the requested job
  """
    jobId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)