from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
from googlecloudsdk.command_lib.kms import flags as kms_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs as presentation_specs_lib
def _GetCryptoKeyVersionResourceSpec():
    return concepts.ResourceSpec(kms_flags.CRYPTO_KEY_VERSION_COLLECTION, resource_name='CryptoKeyVersion', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=concepts.ResourceParameterAttributeConfig(name='location', help_text='The location of the {resource}.'), keyRingsId=concepts.ResourceParameterAttributeConfig(name='keyring', help_text='The keyring of the {resource}.'), cryptoKeysId=concepts.ResourceParameterAttributeConfig(name='key', help_text='The key of the {resource}.'), cryptoKeyVersionsId=concepts.ResourceParameterAttributeConfig(name='version', help_text='The key version of the {resource}.'))