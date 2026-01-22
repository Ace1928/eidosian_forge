from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from argcomplete.completers import DirectoriesCompleter
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.eventarc import flags as eventarc_flags
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def AddKMSKeyFlags(parser):
    """Adds flags for configuring the CMEK key."""
    kmskey_group = parser.add_group(mutex=True)
    kmskey_group.add_argument('--kms-key', type=arg_parsers.RegexpValidator(_KMS_KEY_NAME_PATTERN, _KMS_KEY_NAME_ERROR), help='        Sets the user managed KMS crypto key used to encrypt the Cloud Function\n        and its resources.\n\n        The KMS crypto key name should match the pattern\n        `projects/${PROJECT}/locations/${LOCATION}/keyRings/${KEYRING}/cryptoKeys/${CRYPTOKEY}`\n        where ${PROJECT} is the project, ${LOCATION} is the location of the key\n        ring, and ${KEYRING} is the key ring that contains the ${CRYPTOKEY}\n        crypto key.\n\n        If this flag is set, then a Docker repository created in Artifact\n        Registry must be specified using the `--docker-repository` flag and the\n        repository must be encrypted using the `same` KMS key.\n      ')
    kmskey_group.add_argument('--clear-kms-key', action='store_true', help='        Clears the KMS crypto key used to encrypt the function.\n      ')