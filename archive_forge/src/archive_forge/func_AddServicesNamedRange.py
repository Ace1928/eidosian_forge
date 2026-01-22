from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddServicesNamedRange(parser):
    """Adds --services-named-range flag."""
    parser.add_argument('--services-named-range', help='The name of the existing secondary range in the clusters subnetwork to use for service ClusterIPs. Alternatively, `--services_cidr_block` can be used to automatically create a GKE-managed one.')