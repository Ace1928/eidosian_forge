from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsInstancesListRequest(_messages.Message):
    """A ApigeeOrganizationsInstancesListRequest object.

  Fields:
    pageSize: Maximum number of instances to return. Defaults to 25.
    pageToken: Page token, returned from a previous ListInstances call, that
      you can use to retrieve the next page of content.
    parent: Required. Name of the organization. Use the following structure in
      your request: `organizations/{org}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)