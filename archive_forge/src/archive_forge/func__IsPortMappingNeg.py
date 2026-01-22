from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.apis import arg_utils
def _IsPortMappingNeg(self, network_endpoint_type, client_port_mapping_mode):
    """Checks if the NEG in the request is a Port Mapping NEG."""
    if not client_port_mapping_mode:
        return False
    endpoint_type_enum = self.messages.NetworkEndpointGroup.NetworkEndpointTypeValueValuesEnum
    endpoint_type_enum_value = arg_utils.ChoiceToEnum(network_endpoint_type, endpoint_type_enum)
    client_port_mapping_mode_enum = self.messages.NetworkEndpointGroup.ClientPortMappingModeValueValuesEnum
    client_port_mapping_mode_enum_value = arg_utils.ChoiceToEnum(client_port_mapping_mode, client_port_mapping_mode_enum)
    return endpoint_type_enum_value == endpoint_type_enum.GCE_VM_IP_PORT and client_port_mapping_mode_enum_value == client_port_mapping_mode_enum.CLIENT_PORT_PER_ENDPOINT