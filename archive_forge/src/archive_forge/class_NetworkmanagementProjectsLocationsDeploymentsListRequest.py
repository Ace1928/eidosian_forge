from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsDeploymentsListRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsDeploymentsListRequest object.

  Fields:
    pageSize: Maximum number of Deployments to return per call. If
      unspecified, will default to 10. The maximum value is 1000; values above
      1000 will be coerced to 1000.
    pageToken: The value returned by the last `ListDeploymentsResponse`
      Indicates that this is a continuation of a prior `ListDeployments` call,
      and that the system should return the next page of data.
    parent: Required. The project and location from which the Deployments
      should be listed, specified in the format
      `projects/{project}/locations/{location}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)