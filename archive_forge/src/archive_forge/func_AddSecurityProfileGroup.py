from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddSecurityProfileGroup(parser):
    """Adds security profile group to this rule."""
    parser.add_argument('--security-profile-group', metavar='SECURITY_PROFILE_GROUP', required=False, help='An org-based security profile group to be used with apply_security_profile_group action. Allowed formats are: a) http(s)://<namespace>/<api>/organizations/<org_id>/locations/global/securityProfileGroups/<profile> b) (//)<namespace>/organizations/<org_id>/locations/global/securityProfileGroups/<profile> c) <profile>. In case "c" gCloud CLI will create a reference matching format "a", but to make it work CLOUDSDK_API_ENDPOINT_OVERRIDES_NETWORKSECURITY property must be set. In order to set this property, please run the command gcloud config set api_endpoint_overrides/networksecurity https://<namespace>/.')