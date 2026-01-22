from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobReference(_messages.Message):
    """A JobReference object.

  Fields:
    jobId: [Required] The ID of the job. The ID must contain only letters
      (a-z, A-Z), numbers (0-9), underscores (_), or dashes (-). The maximum
      length is 1,024 characters.
    projectId: [Required] The ID of the project containing this job.
  """
    jobId = _messages.StringField(1)
    projectId = _messages.StringField(2)