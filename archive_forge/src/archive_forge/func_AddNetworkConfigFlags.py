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
def AddNetworkConfigFlags(parser):
    """Adds flags related to the network config for the node pool.

  Args:
    parser: A given parser.
  """
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pod-ipv4-range', metavar='NAME', help="\nSet the pod range to be used as the source for pod IPs for the pods in this node\npool. NAME must be the name of an existing subnetwork secondary range in the\nsubnetwork for this cluster.\n\nMust be used in VPC native clusters. Cannot be used with\n`--create-ipv4-pod-range`.\n\nExamples:\n\nSpecify a pod range called ``other-range''\n\n  $ {command} --pod-ipv4-range other-range\n")
    group.add_argument('--create-pod-ipv4-range', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='\nCreate a new pod range for the node pool. The name and range of the\npod range can be customized via optional ``name\'\' and ``range\'\' keys.\n\n``name\'\' specifies the name of the secondary range to be created.\n\n``range\'\' specifies the IP range for the new secondary range. This can either\nbe a netmask size (e.g. "/20") or a CIDR range (e.g. "10.0.0.0/20").\nIf a netmask size is specified, the IP is automatically taken from the\nfree space in the cluster\'s network.\n\nMust be used in VPC native clusters. Can not be used in conjunction with the\n`--pod-ipv4-range` option.\n\nExamples:\n\nCreate a new pod range with a default name and size.\n\n  $ {command} --create-pod-ipv4-range ""\n\nCreate a new pod range named ``my-range\'\' with netmask of size ``21\'\'.\n\n  $ {command} --create-pod-ipv4-range name=my-range,range=/21\n\nCreate a new pod range with a default name with the primary range of\n``10.100.0.0/16\'\'.\n\n  $ {command} --create-pod-ipv4-range range=10.100.0.0/16\n\nCreate a new pod range with the name ``my-range\'\' with a default range.\n\n  $ {command} --create-pod-ipv4-range name=my-range\n\nMust be used in VPC native clusters. Can not be used in conjunction with the\n`--pod-ipv4-range` option.\n')