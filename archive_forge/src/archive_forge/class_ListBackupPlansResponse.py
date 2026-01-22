from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBackupPlansResponse(_messages.Message):
    """The response message for getting a list of `BackupPlan`.

  Fields:
    backupPlans: The list of `BackupPlans` in the project for the specified
      location. If the `{location}` value in the request is "-", the response
      contains a list of resources from all locations. In case any location is
      unreachable, the response will only return backup plans in reachable
      locations and the 'unreachable' field will be populated with a list of
      unreachable locations. BackupPlan
    nextPageToken: A token which may be sent as page_token in a subsequent
      `ListBackupPlans` call to retrieve the next page of results. If this
      field is omitted or empty, then there are no more results to return.
    unreachable: Locations that could not be reached.
  """
    backupPlans = _messages.MessageField('BackupPlan', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)