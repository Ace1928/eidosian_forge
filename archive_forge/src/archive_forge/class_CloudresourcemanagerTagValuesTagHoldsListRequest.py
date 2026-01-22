from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagValuesTagHoldsListRequest(_messages.Message):
    """A CloudresourcemanagerTagValuesTagHoldsListRequest object.

  Fields:
    filter: Optional. Criteria used to select a subset of TagHolds parented by
      the TagValue to return. This field follows the syntax defined by
      aip.dev/160; the `holder` and `origin` fields are supported for
      filtering. Currently only `AND` syntax is supported. Some example
      queries are: * `holder =
      //compute.googleapis.com/compute/projects/myproject/regions/us-
      east-1/instanceGroupManagers/instance-group` * `origin = 35678234` *
      `holder =
      //compute.googleapis.com/compute/projects/myproject/regions/us-
      east-1/instanceGroupManagers/instance-group AND origin = 35678234`
    pageSize: Optional. The maximum number of TagHolds to return in the
      response. The server allows a maximum of 300 TagHolds to return. If
      unspecified, the server will use 100 as the default.
    pageToken: Optional. A pagination token returned from a previous call to
      `ListTagHolds` that indicates where this listing should continue from.
    parent: Required. The resource name of the parent TagValue. Must be of the
      form: `tagValues/{tag-value-id}`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)