from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def GetFindingIdFromName(finding_name):
    """Gets a finding id from the full resource name."""
    resource_pattern = re.compile('(organizations|projects|folders)/.*/sources/[0-9-]+/findings/[a-zA-Z0-9]+$')
    region_resource_pattern = re.compile('(organizations|projects|folders)/.*/sources/[0-9-]+/locations/.*/findings/[a-zA-Z0-9]+$')
    if not resource_pattern.match(finding_name) and (not region_resource_pattern.match(finding_name)):
        raise errors.InvalidSCCInputError('When providing a full resource path, it must include the pattern organizations/[0-9]+/sources/[0-9-]+/findings/[a-zA-Z0-9]+.')
    list_finding_components = finding_name.split('/')
    return list_finding_components[len(list_finding_components) - 1]