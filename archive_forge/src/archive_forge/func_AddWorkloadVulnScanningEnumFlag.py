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
def AddWorkloadVulnScanningEnumFlag(parser):
    """Adds Kubernetes Security Posture's Workload Vulnerability Scanning flag to the parser."""
    choices = ['disabled', 'standard', 'enterprise']
    parser.add_argument('--workload-vulnerability-scanning', choices=choices, default=None, hidden=False, help=textwrap.dedent("      Sets the mode of the Kubernetes security posture API's workload\n      vulnerability scanning.\n\n      To enable Advanced vulnerability insights mode explicitly set the flag to\n      `--workload-vulnerability-scanning=enterprise`.\n\n      To enable in standard mode explicitly set the flag to\n      `--workload-vulnerability-scanning=standard`.\n\n      To disable in an existing cluster, explicitly set the flag to\n      `--workload-vulnerability-scanning=disabled`.\n      "))