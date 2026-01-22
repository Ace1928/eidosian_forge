from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _GetServerTlsPolicyResourceSpec(region_fallthrough):
    return concepts.ResourceSpec('networksecurity.projects.locations.serverTlsPolicies', resource_name='server_tls_policy', serverTlsPoliciesId=_ServerTlsPolicyAttributeConfig(), locationsId=_LocationAttributeConfig(region_fallthrough), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)