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
def AddConvertToStandardFlag(parser, hidden=True):
    """Adds --convert-to-standard flag to parser."""
    help_text = 'Convert the given Autopilot cluster to Standard. It can also be used to rollback\nthe Autopilot conversion during or after workload migration.\n'
    parser.add_argument('--convert-to-standard', default=None, help=help_text, action='store_true', hidden=hidden)