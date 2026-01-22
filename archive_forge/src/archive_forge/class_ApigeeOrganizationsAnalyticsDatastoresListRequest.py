from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAnalyticsDatastoresListRequest(_messages.Message):
    """A ApigeeOrganizationsAnalyticsDatastoresListRequest object.

  Fields:
    parent: Required. The parent organization name. Must be of the form
      `organizations/{org}`.
    targetType: Optional. TargetType is used to fetch all Datastores that
      match the type
  """
    parent = _messages.StringField(1, required=True)
    targetType = _messages.StringField(2)