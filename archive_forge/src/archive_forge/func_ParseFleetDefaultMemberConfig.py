from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def ParseFleetDefaultMemberConfig(yaml_config, msg):
    """Parses the ASM Fleet Default MEmber Config from a yaml file.

  Args:
    yaml_config: object containing arguments passed as flags with the command
    msg: The gkehub messages package.

  Returns:
    member_config: The Membership spec configuration
  """
    if len(yaml_config.data) != 1:
        raise exceptions.Error('Input config file must contain one YAML document.')
    config = yaml_config.data[0]
    management = config.GetManagement()
    if management is None:
        raise exceptions.Error('Missing required field .management')
    member_config = msg.ServiceMeshMembershipSpec()
    if management == 'automatic':
        member_config.management = msg.ServiceMeshMembershipSpec.ManagementValueValuesEnum('MANAGEMENT_AUTOMATIC')
    elif management == 'manual':
        member_config.management = msg.ServiceMeshMembershipSpec.ManagementValueValuesEnum('MANAGEMENT_MANUAL')
    elif management is None or management == 'unspecified':
        member_config.management = msg.ServiceMeshMembershipSpec.ManagementValueValuesEnum('MANAGEMENT_UNSPECIFIED')
    else:
        status_msg = 'management [{}] is not supported.'.format(management)
        log.status.Print(status_msg)
    return member_config