from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetCmekKeyResourceSpec(resource_name='cmek-key'):
    return concepts.ResourceSpec('cloudkms.projects.locations.keyRings.cryptoKeys', resource_name=resource_name, api_version='v1', cryptoKeysId=CmekKeyAttributeConfig(), keyRingsId=CmekKeyringAttributeConfig(), locationsId=RegionAttributeConfig(), projectsId=CmekProjectAttributeConfig(), disable_auto_completers=False)