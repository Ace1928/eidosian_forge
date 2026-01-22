from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityOrganizationsLocationsSecurityProfilesCreateRequest(_messages.Message):
    """A NetworksecurityOrganizationsLocationsSecurityProfilesCreateRequest
  object.

  Fields:
    parent: Required. The parent resource of the SecurityProfile. Must be in
      the format `projects|organizations/*/locations/{location}`.
    securityProfile: A SecurityProfile resource to be passed as the request
      body.
    securityProfileId: Required. Short name of the SecurityProfile resource to
      be created. This value should be 1-63 characters long, containing only
      letters, numbers, hyphens, and underscores, and should not start with a
      number. E.g. "security_profile1".
  """
    parent = _messages.StringField(1, required=True)
    securityProfile = _messages.MessageField('SecurityProfile', 2)
    securityProfileId = _messages.StringField(3)