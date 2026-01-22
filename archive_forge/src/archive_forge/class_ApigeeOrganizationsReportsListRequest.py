from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsReportsListRequest(_messages.Message):
    """A ApigeeOrganizationsReportsListRequest object.

  Fields:
    expand: Set to 'true' to get expanded details about each custom report.
    parent: Required. The parent organization name under which the API product
      will be listed `organizations/{organization_id}/reports`
  """
    expand = _messages.BooleanField(1)
    parent = _messages.StringField(2, required=True)