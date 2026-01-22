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
def AddShieldedInstanceFlags(parser):
    """Adds Shielded Instance flags to the given parser."""
    secure_boot_help = '      The instance will boot with secure boot enabled.\n      '
    parser.add_argument('--shielded-secure-boot', default=None, action='store_true', help=secure_boot_help)
    integrity_monitoring_help = '      Enables monitoring and attestation of the boot integrity of the\n      instance. The attestation is performed against the integrity policy\n      baseline. This baseline is initially derived from the implicitly\n      trusted boot image when the instance is created.\n      '
    parser.add_argument('--shielded-integrity-monitoring', default=None, action='store_true', help=integrity_monitoring_help)