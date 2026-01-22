from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ThreatPrevention(base.Group):
    """Manage Security Profiles - Threat Prevention Profile.

  Manage Security Profiles - Threat Prevention Profile.

  ## EXAMPLES

  To create a Security Profile with the name `my-security-profile` which
  includes location as global or region specified and organization ID, optional
  description as `New Security Profile`, run:

    $ {command} create my-security-profile  --description="New Security Profile"

  To add an override, run:

    $ {command} add-override my-security-profile --severities=MEDIUM
    --action=ALLOW

    `my-security-profile` is the name of the Security Profile in the
    format organizations/{organizationID}/locations/{location}/securityProfiles/
    {security_profile_id} where organizationID is the organization ID to which
    the changes should apply, location either global or region specified and
    security_profile_id the Security Profile Identifier.

  To update an override, run:

    $ {command} update-override my-security-profile --severities=MEDIUM
    --action=ALLOW

    `my-security-profile` is the name of the Security Profile in the
    format organizations/{organizationID}/locations/{location}/securityProfiles/
    {security_profile_id} where organizationID is the organization ID to which
    the changes should apply, location either global or region specified and
    security_profile_id the Security Profile Identifier.

  To list overrides, run:

    $ {command} list-overrides my-security-profile

    `my-security-profile` is the name of the Security Profile in the
    format organizations/{organizationID}/locations/{location}/securityProfiles/
    {security_profile_id} where organizationID is the organization ID to which
    the changes should apply, location either global or region specified and
    security_profile_id the Security Profile Identifier.

  To delete an override, run:

    $ {command} delete-override my-security-profile --severities=MEDIUM

    `my-security-profile` is the name of the Security Profile in the
    format organizations/{organizationID}/locations/{location}/securityProfiles/
    {security_profile_id} where organizationID is the organization ID to which
    the changes should apply, location either global or region specified and
    security_profile_id the Security Profile Identifier.

  To list Security Profiles in specified location and organization, run:

    $ {command} list --location=global

  To delete a Security Profile called `my-security-profile` which includes
  location as global or region specified and organization ID, run:

      $ {command} delete my-security-profile
  """
    category = base.NETWORK_SECURITY_CATEGORY