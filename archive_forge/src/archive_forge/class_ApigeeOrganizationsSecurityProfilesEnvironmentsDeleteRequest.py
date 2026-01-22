from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesEnvironmentsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesEnvironmentsDeleteRequest object.

  Fields:
    name: Required. The name of the environment attachment to delete. Format:
      organizations/{org}/securityProfiles/{profile}/environments/{env}
  """
    name = _messages.StringField(1, required=True)