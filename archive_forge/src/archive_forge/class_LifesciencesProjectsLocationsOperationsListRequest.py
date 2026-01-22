from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LifesciencesProjectsLocationsOperationsListRequest(_messages.Message):
    """A LifesciencesProjectsLocationsOperationsListRequest object.

  Fields:
    filter: A string for filtering Operations. The following filter fields are
      supported: * createTime: The time this job was created * events: The set
      of event (names) that have occurred while running the pipeline. The :
      operator can be used to determine if a particular event has occurred. *
      error: If the pipeline is running, this value is NULL. Once the pipeline
      finishes, the value is the standard Google error code. * labels.key or
      labels."key with space" where key is a label key. * done: If the
      pipeline is running, this value is false. Once the pipeline finishes,
      the value is true.
    name: The name of the operation's parent resource.
    pageSize: The maximum number of results to return. The maximum value is
      256.
    pageToken: The standard list page token.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)