from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityReport(_messages.Message):
    """SecurityReport saves all the information about the created security
  report.

  Fields:
    created: Creation time of the query.
    displayName: Display Name specified by the user.
    envgroupHostname: Hostname is available only when query is executed at
      host level.
    error: Error is set when query fails.
    executionTime: ExecutionTime is available only after the query is
      completed.
    queryParams: Contains information like metrics, dimenstions etc of the
      Security Report.
    reportDefinitionId: Report Definition ID.
    result: Result is available only after the query is completed.
    resultFileSize: ResultFileSize is available only after the query is
      completed.
    resultRows: ResultRows is available only after the query is completed.
    self: Self link of the query. Example: `/organizations/myorg/environments/
      myenv/securityReports/9cfc0d85-0f30-46d6-ae6f-318d0cb961bd` or following
      format if query is running at host level: `/organizations/myorg/hostSecu
      rityReports/9cfc0d85-0f30-46d6-ae6f-318d0cb961bd`
    state: Query state could be "enqueued", "running", "completed", "expired"
      and "failed".
    updated: Output only. Last updated timestamp for the query.
  """
    created = _messages.StringField(1)
    displayName = _messages.StringField(2)
    envgroupHostname = _messages.StringField(3)
    error = _messages.StringField(4)
    executionTime = _messages.StringField(5)
    queryParams = _messages.MessageField('GoogleCloudApigeeV1SecurityReportMetadata', 6)
    reportDefinitionId = _messages.StringField(7)
    result = _messages.MessageField('GoogleCloudApigeeV1SecurityReportResultMetadata', 8)
    resultFileSize = _messages.StringField(9)
    resultRows = _messages.IntegerField(10)
    self = _messages.StringField(11)
    state = _messages.StringField(12)
    updated = _messages.StringField(13)