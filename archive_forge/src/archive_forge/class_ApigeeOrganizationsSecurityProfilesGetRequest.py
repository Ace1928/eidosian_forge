from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesGetRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesGetRequest object.

  Fields:
    name: Required. Security profile in the following format:
      `organizations/{org}/securityProfiles/{profile}'. Profile may optionally
      contain revision ID. If revision ID is not provided, the response will
      contain latest revision by default. Example:
      organizations/testOrg/securityProfiles/testProfile@5
  """
    name = _messages.StringField(1, required=True)