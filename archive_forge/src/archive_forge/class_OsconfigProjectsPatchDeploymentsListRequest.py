from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchDeploymentsListRequest(_messages.Message):
    """A OsconfigProjectsPatchDeploymentsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of patch deployments to return.
      Default is 100.
    pageToken: Optional. A pagination token returned from a previous call to
      ListPatchDeployments that indicates where this listing should continue
      from.
    parent: Required. The resource name of the parent in the form
      `projects/*`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)