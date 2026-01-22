from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobsSummary(_messages.Message):
    """Data related to Jobs page summary

  Fields:
    activeJobs: Number of active jobs
    applicationId: Spark Application Id
    attempts: Attempts info
    completedJobs: Number of completed jobs
    failedJobs: Number of failed jobs
    schedulingMode: Spark Scheduling mode
  """
    activeJobs = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    applicationId = _messages.StringField(2)
    attempts = _messages.MessageField('ApplicationAttemptInfo', 3, repeated=True)
    completedJobs = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    failedJobs = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    schedulingMode = _messages.StringField(6)