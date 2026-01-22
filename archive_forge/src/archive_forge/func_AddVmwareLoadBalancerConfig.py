from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddVmwareLoadBalancerConfig(parser: parser_arguments.ArgumentInterceptor, for_update=False):
    """Adds a command group to set the load balancer config.

  Args:
    parser: The argparse parser to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    required = False if for_update else True
    vmware_load_balancer_config_group = parser.add_group(help='Anthos on VMware cluster load balancer configurations', required=required)
    _AddVmwareVipConfig(vmware_load_balancer_config_group, for_update=for_update)
    lb_config_mutex_group = vmware_load_balancer_config_group.add_group(mutex=True, help='Populate one of the load balancers.', required=required)
    _AddMetalLbConfig(lb_config_mutex_group)
    _AddF5Config(lb_config_mutex_group, for_update=for_update)
    _AddManualLbConfig(lb_config_mutex_group, for_update=for_update)