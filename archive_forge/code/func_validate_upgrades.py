from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
def validate_upgrades(upgrades):
    """Validate the string `upgrades`.

  Args:
    upgrades: a string.

  Raises: Error, if upgrades is invalid.
  """
    legal_fields = [utils.UPGRADES_AUTO, utils.UPGRADES_MANUAL, utils.UPGRADES_EMPTY]
    valid_values = ' '.join((f"'{field}'" for field in legal_fields))
    if upgrades not in legal_fields:
        raise exceptions.Error('The valid values of field .spec.{} are: {}'.format(utils.UPGRADES, valid_values))