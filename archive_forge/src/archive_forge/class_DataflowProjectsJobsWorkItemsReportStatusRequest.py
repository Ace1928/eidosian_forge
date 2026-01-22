from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsJobsWorkItemsReportStatusRequest(_messages.Message):
    """A DataflowProjectsJobsWorkItemsReportStatusRequest object.

  Fields:
    jobId: The job which the WorkItem is part of.
    projectId: The project which owns the WorkItem's job.
    reportWorkItemStatusRequest: A ReportWorkItemStatusRequest resource to be
      passed as the request body.
  """
    jobId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    reportWorkItemStatusRequest = _messages.MessageField('ReportWorkItemStatusRequest', 3)