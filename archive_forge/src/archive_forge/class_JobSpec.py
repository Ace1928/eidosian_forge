from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobSpec(_messages.Message):
    """JobSpec describes how the job will look.

  Fields:
    template: Optional. Describes the execution that will be created when
      running a job.
  """
    template = _messages.MessageField('ExecutionTemplateSpec', 1)