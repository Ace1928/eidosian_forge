from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetPolicyResourceSpec(api_version):
    return concepts.ResourceSpec('dns.policies', api_version=api_version, resource_name='policy', policy=PolicyAttributeConfig(api_version=api_version), project=ProjectAttributeConfig())