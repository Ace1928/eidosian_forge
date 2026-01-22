from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddBGPLBAsn(bgp_lb_config_group, is_update=False):
    """Adds flags for ASN used by BGP LB load balancer of the cluster.

  Args:
   bgp_lb_config_group: The parent group to add the flags to.
   is_update: bool, whether the flag is for update command or not.
  """
    bgp_lb_config_group.add_argument('--bgp-lb-asn', required=not is_update, help='BGP autonomous system number (ASN) of the cluster.', type=int)