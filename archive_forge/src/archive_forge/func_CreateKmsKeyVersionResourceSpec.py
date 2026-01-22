from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import locations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.kms import resource_args as kms_args
from googlecloudsdk.command_lib.privateca import completers as privateca_completers
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def CreateKmsKeyVersionResourceSpec():
    """Creates a resource spec for a KMS CryptoKeyVersion.

  Defaults to the location and project of the CA, specified through flags or
  properties.

  Returns:
    A concepts.ResourceSpec for a CryptoKeyVersion.
  """
    return concepts.ResourceSpec('cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions', resource_name='key version', cryptoKeyVersionsId=kms_args.KeyVersionAttributeConfig(kms_prefix=True), cryptoKeysId=kms_args.KeyAttributeConfig(kms_prefix=True), keyRingsId=kms_args.KeyringAttributeConfig(kms_prefix=True), locationsId=LocationAttributeConfig('kms-location', [deps.ArgFallthrough('location'), LOCATION_PROPERTY_FALLTHROUGH]), projectsId=ProjectAttributeConfig('kms-project', [deps.ArgFallthrough('project'), PROJECT_PROPERTY_FALLTHROUGH]))