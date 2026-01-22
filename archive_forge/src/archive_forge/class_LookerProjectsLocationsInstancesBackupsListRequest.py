from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesBackupsListRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesBackupsListRequest object.

  Fields:
    orderBy: Sort results. Default order is "create_time desc". Other
      supported fields are "state" and "expire_time".
      https://google.aip.dev/132#ordering
    pageSize: The maximum number of instances to return.
    pageToken: A page token received from a previous ListInstances request.
    parent: Required. Format:
      projects/{project}/locations/{location}/instances/{instance}.
  """
    orderBy = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)