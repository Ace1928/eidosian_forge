from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSitesApicategoriesGetRequest(_messages.Message):
    """A ApigeeOrganizationsSitesApicategoriesGetRequest object.

  Fields:
    name: Required. Name of the category. Use the following structure in your
      request: `organizations/{org}/sites/{site}/apicategories/{apicategory}`
  """
    name = _messages.StringField(1, required=True)