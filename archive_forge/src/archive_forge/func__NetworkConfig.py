from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _NetworkConfig(alloydb_messages, assign_inbound_public_ip=None, authorized_external_networks=None):
    """Generates the instance network config for the instance.

  Args:
    alloydb_messages: module, Message module for the API client.
    assign_inbound_public_ip: string, whether or not to enable Public-IP.
    authorized_external_networks: list, list of external networks authorized to
      access the instance.

  Returns:
    alloydb_messages.NetworkConfig
  """
    should_generate_config = any([assign_inbound_public_ip, authorized_external_networks is not None])
    if not should_generate_config:
        return None
    instance_network_config = alloydb_messages.InstanceNetworkConfig()
    if assign_inbound_public_ip:
        instance_network_config.enablePublicIp = _ParseAssignInboundPublicIp(assign_inbound_public_ip)
    if authorized_external_networks is not None:
        if assign_inbound_public_ip is not None and (not instance_network_config.enablePublicIp):
            raise DetailedArgumentError("Cannot update an instance's authorized networks and disable Public-IP. You must do one or the other. Note, that disabling Public-IP will clear the list of authorized networks.")
        instance_network_config.authorizedExternalNetworks = _ParseAuthorizedExternalNetworks(alloydb_messages, authorized_external_networks, instance_network_config.enablePublicIp)
    return instance_network_config