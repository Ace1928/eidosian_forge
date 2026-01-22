from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
def _build_monitoring_msg(spec_monitoring, msg):
    """Build PolicyControllerMonitoring message from the parsed spec.

  Args:
    spec_monitoring: dict, The monitoring data loaded from the
      config-management.yaml given by user.
    msg: The Hub messages package.

  Returns:
    monitoring: The Policy Controller Monitoring configuration for
    MembershipConfigs, filled in the data parsed from
    configmanagement.spec.policyController.monitoring
  Raises: Error, if Policy Controller Monitoring Backend is not recognized
  """
    backends = spec_monitoring.get('backends', [])
    if not backends:
        return None
    converter = constants.monitoring_backend_converter(msg)

    def convert(backend):
        result = converter.get(backend.lower())
        if not result:
            raise exceptions.Error('policyController.monitoring.backend {} is not recognized'.format(backend))
        return result
    monitoring_backends = [convert(backend) for backend in backends]
    monitoring = msg.ConfigManagementPolicyControllerMonitoring()
    monitoring.backends = monitoring_backends
    return monitoring