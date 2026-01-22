from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddUpdateNegArgsToParser(parser, support_ipv6=False, support_port_mapping_neg=False):
    """Adds flags for updating a network endpoint group to the parser."""
    endpoint_group = parser.add_group(mutex=True, required=True, help='These flags can be specified multiple times to add/remove multiple endpoints.')
    endpoint_spec = {'instance': str, 'ip': str, 'port': int, 'fqdn': str}
    if support_ipv6:
        endpoint_spec['ipv6'] = str
    if support_port_mapping_neg:
        endpoint_spec['client-port'] = int
    _AddAddEndpoint(endpoint_group, endpoint_spec, support_ipv6, support_port_mapping_neg)
    _AddRemoveEndpoint(endpoint_group, endpoint_spec, support_ipv6, support_port_mapping_neg)