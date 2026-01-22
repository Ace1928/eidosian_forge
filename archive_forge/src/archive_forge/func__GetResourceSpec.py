from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def _GetResourceSpec():
    return concepts.ResourceSpec('accesscontextmanager.accessPolicies.authorizedOrgsDescs', resource_name='authorized_orgs_desc', accessPoliciesId=policies.GetAttributeConfig(), authorizedOrgsDescsId=_GetAttributeConfig())