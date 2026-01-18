from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
def upgradesFromStr(self, upgrades):
    """Convert the string `upgrades` to an enum in the ACM Fleet Feature API.

    Args:
      upgrades: a string.

    Returns:
      an enum represent the field `management` in the ACM Fleet Feature API.
    """
    if upgrades == utils.UPGRADES_AUTO:
        return self.messages.ConfigManagementMembershipSpec.ManagementValueValuesEnum(utils.MANAGEMENT_AUTOMATIC)
    else:
        return self.messages.ConfigManagementMembershipSpec.ManagementValueValuesEnum(utils.MANAGEMENT_MANUAL)