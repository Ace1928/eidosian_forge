from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
from googlecloudsdk.command_lib.scc.hooks import GetOrganization
from googlecloudsdk.command_lib.scc.hooks import GetParentFromResourceName
from googlecloudsdk.command_lib.scc.util import GetParentFromPositionalArguments
def _GetAssetName(args):
    """Prepares asset relative path using organization and asset."""
    resource_pattern = re.compile('organizations/[0-9]+/assets/[0-9]+')
    id_pattern = re.compile('[0-9]+')
    if not resource_pattern.match(args.asset) and (not id_pattern.match(args.asset)):
        raise InvalidSCCInputError('Asset must match either organizations/[0-9]+/assets/[0-9]+ or [0-9]+.')
    if resource_pattern.match(args.asset):
        return args.asset
    return GetOrganization(args) + '/assets/' + args.asset