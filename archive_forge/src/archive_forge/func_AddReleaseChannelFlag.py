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
def AddReleaseChannelFlag(parser, is_update=False, autopilot=False, hidden=False):
    """Adds a --release-channel flag to the given parser."""
    short_text = 'Release channel a cluster is subscribed to.\n\nIf left unspecified and a version is specified, the cluster is enrolled in the\nmost mature release channel where the version is available (first checking\nSTABLE, then REGULAR, and finally RAPID). Otherwise, if no release channel and\nno version is specified, the cluster is enrolled in the REGULAR channel with\nits default version.\n'
    if is_update:
        short_text = 'Subscribe or unsubscribe this cluster to a release channel.\n\n'
    help_text = short_text + 'When a cluster is subscribed to a release channel, Google maintains both the\nmaster version and the node version. Node auto-upgrade is enabled by default\nfor release channel clusters and can be controlled via [upgrade-scope\nexclusions](https://cloud.google.com/kubernetes-engine/docs/concepts/maintenance-windows-and-exclusions#scope_of_maintenance_to_exclude).\n'
    choices = {'rapid': "'rapid' channel is offered on an early access basis for customers who want\nto test new releases.\n\nWARNING: Versions available in the 'rapid' channel may be subject to\nunresolved issues with no known workaround and are not subject to any\nSLAs.\n", 'regular': "Clusters subscribed to 'regular' receive versions that are considered GA\nquality. 'regular' is intended for production users who want to take\nadvantage of new features.", 'stable': "Clusters subscribed to 'stable' receive versions that are known to be\nstable and reliable in production."}
    if not autopilot:
        choices.update({'None': "Use 'None' to opt-out of any release channel.\n"})
    return parser.add_argument('--release-channel', metavar='CHANNEL', choices=choices, help=help_text, hidden=hidden)