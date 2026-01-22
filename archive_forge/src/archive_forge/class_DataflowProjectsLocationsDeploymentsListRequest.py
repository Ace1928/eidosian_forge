from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsDeploymentsListRequest(_messages.Message):
    """A DataflowProjectsLocationsDeploymentsListRequest object.

  Fields:
    pageSize: The maximum number of deployments to return. The service may
      return fewer than this value. If unspecified, at most 50 deployments
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: A page token, received from a previous `ListDeployments` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListDeployments` must match the call that
      provided the page token.
    parent: Required. The `location`, which owns this collection of
      deployments. Format: projects/{project}/locations/{location}
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)