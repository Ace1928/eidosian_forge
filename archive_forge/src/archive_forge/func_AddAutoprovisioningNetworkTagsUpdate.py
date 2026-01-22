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
def AddAutoprovisioningNetworkTagsUpdate(parser):
    """Adds a --autoprovisioning-network-tags flag to the given parser."""
    help_text = 'Replaces the user specified Compute Engine tags on all nodes in all the existing\nauto-provisioned node pools in the Standard cluster or the Autopilot with the given tags (comma separated).\n\nExamples:\n\n  $ {command} example-cluster --autoprovisioning-network-tags=tag1,tag2\n\nNew nodes in auto-provisioned node pools, including ones created by resize or recreate, will have these tags\non the Compute Engine API instance object and these tags can be used in\nfirewall rules.\nSee https://cloud.google.com/sdk/gcloud/reference/compute/firewall-rules/create\nfor examples.\n'
    parser.add_argument('--autoprovisioning-network-tags', metavar='TAGS', type=arg_parsers.ArgList(), help=help_text)