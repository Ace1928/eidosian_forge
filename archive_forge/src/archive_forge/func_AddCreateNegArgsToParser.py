from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddCreateNegArgsToParser(parser, support_neg_type=False, support_serverless_deployment=False, support_port_mapping_neg=False):
    """Adds flags for creating a network endpoint group to the parser."""
    _AddNetworkEndpointGroupType(parser, support_neg_type)
    _AddNetworkEndpointType(parser)
    _AddNetwork(parser)
    _AddSubnet(parser)
    _AddDefaultPort(parser)
    _AddServerlessRoutingInfo(parser, support_serverless_deployment)
    _AddL7pscRoutingInfo(parser)
    if support_port_mapping_neg:
        _AddPortMappingInfo(parser)