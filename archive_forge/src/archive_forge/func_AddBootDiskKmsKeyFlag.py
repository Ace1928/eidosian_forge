from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddBootDiskKmsKeyFlag(parser, suppressed=False, help_text=''):
    """Adds a --boot-disk-kms-key flag to the given parser."""
    help_text += 'The Customer Managed Encryption Key used to encrypt the boot disk attached\nto each node in the node pool. This should be of the form\nprojects/[KEY_PROJECT_ID]/locations/[LOCATION]/keyRings/[RING_NAME]/cryptoKeys/[KEY_NAME].\nFor more information about protecting resources with Cloud KMS Keys please\nsee:\nhttps://cloud.google.com/compute/docs/disks/customer-managed-encryption'
    parser.add_argument('--boot-disk-kms-key', help=help_text, hidden=suppressed, type=str, default='')