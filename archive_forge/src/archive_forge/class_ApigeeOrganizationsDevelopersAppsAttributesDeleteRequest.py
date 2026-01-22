from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAppsAttributesDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAppsAttributesDeleteRequest object.

  Fields:
    name: Required. Name of the developer app attribute. Use the following
      structure in your request: `organizations/{org}/developers/{developer_em
      ail}/apps/{app}/attributes/{attribute}`
  """
    name = _messages.StringField(1, required=True)