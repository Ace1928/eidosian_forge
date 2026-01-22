from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsGetProjectMappingRequest(_messages.Message):
    """A ApigeeOrganizationsGetProjectMappingRequest object.

  Fields:
    name: Required. Apigee organization name in the following format:
      `organizations/{org}`
  """
    name = _messages.StringField(1, required=True)