from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WriteLogEntriesRequest(_messages.Message):
    """The parameters to WriteLogEntries.

  Messages:
    LabelsValue: Optional. Default labels that are added to the labels field
      of all log entries in entries. If a log entry already has a label with
      the same key as a label in this parameter, then the log entry's label is
      not changed. See LogEntry.

  Fields:
    dryRun: Optional. If true, the request should expect normal response, but
      the entries won't be persisted nor exported. Useful for checking whether
      the logging API endpoints are working properly before sending valuable
      data.
    entries: Required. The log entries to send to Logging. The order of log
      entries in this list does not matter. Values supplied in this method's
      log_name, resource, and labels fields are copied into those log entries
      in this list that do not include values for their corresponding fields.
      For more information, see the LogEntry type.If the timestamp or
      insert_id fields are missing in log entries, then this method supplies
      the current time or a unique identifier, respectively. The supplied
      values are chosen so that, among the log entries that did not supply
      their own values, the entries earlier in the list will sort before the
      entries later in the list. See the entries.list method.Log entries with
      timestamps that are more than the logs retention period
      (https://cloud.google.com/logging/quotas) in the past or more than 24
      hours in the future will not be available when calling entries.list.
      However, those log entries can still be exported with LogSinks
      (https://cloud.google.com/logging/docs/api/tasks/exporting-logs).To
      improve throughput and to avoid exceeding the quota limit
      (https://cloud.google.com/logging/quotas) for calls to entries.write,
      you should try to include several log entries in this list, rather than
      calling this method for each individual log entry.
    labels: Optional. Default labels that are added to the labels field of all
      log entries in entries. If a log entry already has a label with the same
      key as a label in this parameter, then the log entry's label is not
      changed. See LogEntry.
    logName: Optional. A default log resource name that is assigned to all log
      entries in entries that do not specify a value for log_name:
      projects/[PROJECT_ID]/logs/[LOG_ID]
      organizations/[ORGANIZATION_ID]/logs/[LOG_ID]
      billingAccounts/[BILLING_ACCOUNT_ID]/logs/[LOG_ID]
      folders/[FOLDER_ID]/logs/[LOG_ID][LOG_ID] must be URL-encoded. For
      example: "projects/my-project-id/logs/syslog"
      "organizations/123/logs/cloudaudit.googleapis.com%2Factivity" The
      permission logging.logEntries.create is needed on each project,
      organization, billing account, or folder that is receiving new log
      entries, whether the resource is specified in logName or in an
      individual log entry.
    partialSuccess: Optional. Whether a batch's valid entries should be
      written even if some other entry failed due to a permanent error such as
      INVALID_ARGUMENT or PERMISSION_DENIED. If any entry failed, then the
      response status is the response status of one of the failed entries. The
      response will include error details in
      WriteLogEntriesPartialErrors.log_entry_errors keyed by the entries'
      zero-based index in the entries. Failed requests for which no entries
      are written will not include per-entry errors.
    resource: Optional. A default monitored resource object that is assigned
      to all log entries in entries that do not specify a value for resource.
      Example: { "type": "gce_instance", "labels": { "zone": "us-central1-a",
      "instance_id": "00000000000000000000" }} See LogEntry.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Default labels that are added to the labels field of all log
    entries in entries. If a log entry already has a label with the same key
    as a label in this parameter, then the log entry's label is not changed.
    See LogEntry.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    dryRun = _messages.BooleanField(1)
    entries = _messages.MessageField('LogEntry', 2, repeated=True)
    labels = _messages.MessageField('LabelsValue', 3)
    logName = _messages.StringField(4)
    partialSuccess = _messages.BooleanField(5)
    resource = _messages.MessageField('MonitoredResource', 6)