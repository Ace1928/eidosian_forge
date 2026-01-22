from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsArchiveDeploymentsListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsArchiveDeploymentsListRequest object.

  Fields:
    filter: Optional. An optional query used to return a subset of Archive
      Deployments using the semantics defined in https://google.aip.dev/160.
    pageSize: Optional. Maximum number of Archive Deployments to return. If
      unspecified, at most 25 deployments will be returned.
    pageToken: Optional. Page token, returned from a previous
      ListArchiveDeployments call, that you can use to retrieve the next page.
    parent: Required. Name of the Environment for which to list Archive
      Deployments in the format: `organizations/{org}/environments/{env}`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)