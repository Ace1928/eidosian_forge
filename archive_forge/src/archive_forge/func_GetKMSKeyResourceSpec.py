from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetKMSKeyResourceSpec(resource_name='kms-key'):
    return concepts.ResourceSpec('cloudkms.projects.locations.keyRings.cryptoKeys', resource_name=resource_name, api_version='v1', cryptoKeysId=CmekKeyAttributeConfig('kms-key'), keyRingsId=CmekKeyringAttributeConfig('kms-keyring'), locationsId=RegionAttributeConfig(), projectsId=CmekProjectAttributeConfig('kms-project'), disable_auto_completers=False)