from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSitesApicategoriesListRequest(_messages.Message):
    """A ApigeeOrganizationsSitesApicategoriesListRequest object.

  Fields:
    parent: Required. Name of the portal. Use the following structure in your
      request: `organizations/{org}/sites/{site}`
  """
    parent = _messages.StringField(1, required=True)