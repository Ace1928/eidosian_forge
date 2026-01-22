from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetDefaultOrganization():
    """Prepend organizations/ to org if necessary."""
    resource_pattern = re.compile('organizations/[0-9]+')
    id_pattern = re.compile('[0-9]+')
    organization = properties.VALUES.scc.organization.Get()
    if not resource_pattern.match(organization) and (not id_pattern.match(organization)):
        raise errors.InvalidSCCInputError('Organization must match either organizations/[0-9]+ or [0-9]+.')
    if resource_pattern.match(organization):
        return organization
    return 'organizations/' + organization