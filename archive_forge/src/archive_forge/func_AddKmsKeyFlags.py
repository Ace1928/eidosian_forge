from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def AddKmsKeyFlags(parser):
    """Adds flags for configuring the CMEK key.

  Args:
    parser: The flag parser used for the specified command.
  """
    kmskey_group = parser.add_group(mutex=True, hidden=True)
    kmskey_group.add_argument('--kms-key', type=arg_parsers.RegexpValidator(_KEY_NAME_PATTERN, _KEY_NAME_ERROR), help='        Sets the user managed KMS crypto key used to encrypt the new Workflow\n        Revision and the Executions associated with it.\n\n        The KMS crypto key name should match the pattern\n        `projects/${PROJECT}/locations/${LOCATION}/keyRings/${KEYRING}/cryptoKeys/${CRYPTOKEY}`\n        where ${PROJECT} is the project, ${LOCATION} is the location of the key\n        ring, and ${KEYRING} is the key ring that contains the ${CRYPTOKEY}\n        crypto key.\n      ')
    kmskey_group.add_argument('--clear-kms-key', action='store_true', help='        Creates the new Workflow Revision and its associated Executions without\n        the KMS key specified on the previous revision.\n      ')