from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.certificate_manager import certificate_maps
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.certificate_manager import resource_args
from googlecloudsdk.command_lib.certificate_manager import util
from googlecloudsdk.core.resource import resource_transform
def _TransformGclbTargets(targets, undefined=''):
    """Transforms GclbTargets to more compact form.

  It uses following format: IP_1:port_1\\nIP_2:port_2\\n...IP_n:port_n.

  Args:
    targets: GclbTargets API representation.
    undefined: str, value to be returned if no IP:port pair is found.

  Returns:
    String representation to be shown in table view.
  """
    if not targets:
        return undefined
    result = []
    for target in targets:
        ip_configs = resource_transform.GetKeyValue(target, 'ipConfigs', None)
        if ip_configs is None:
            return undefined
        for ip_config in ip_configs:
            ip_address = resource_transform.GetKeyValue(ip_config, 'ipAddress', None)
            ports = resource_transform.GetKeyValue(ip_config, 'ports', None)
            if ip_address is None or ports is None:
                continue
            for port in ports:
                result.append('{}:{}'.format(ip_address, port))
    return '\n'.join(result) if result else undefined