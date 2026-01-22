from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsResourcefilesDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsResourcefilesDeleteRequest object.

  Fields:
    name: Required. ID of the resource file to delete. Must match the regular
      expression: [a-zA-Z0-9:/\\\\!@#$%^&{}\\[\\]()+\\-=,.~'` ]{1,255}
    parent: Required. Name of the environment in the following format:
      `organizations/{org}/environments/{env}`.
    type: Required. Resource file type. {{ resource_file_type }}
  """
    name = _messages.StringField(1, required=True)
    parent = _messages.StringField(2, required=True)
    type = _messages.StringField(3, required=True)