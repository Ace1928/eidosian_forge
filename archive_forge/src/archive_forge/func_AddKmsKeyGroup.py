from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddKmsKeyGroup(parser):
    """Register flags for KMS Key."""
    key_group = parser.add_group(required=True, help="Key resource - The Cloud KMS (Key Management Service) cryptokey that will be used to protect the Looker instance and backups. The 'Looker Service Agent' service account must hold role 'Cloud KMS CryptoKey Encrypter'. The arguments in this group can be used to specify the attributes of this resource.")
    key_group.add_argument('--kms-key', metavar='KMS_KEY', required=True, help='Fully qualified identifier (name) for the key.')