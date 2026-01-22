from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
def _parse_policy_controller(configmanagement, msg):
    """Load PolicyController with the parsed config-management.yaml.

  Args:
    configmanagement: dict, The data loaded from the config-management.yaml
      given by user.
    msg: The Hub messages package.

  Returns:
    policy_controller: The Policy Controller configuration for
    MembershipConfigs, filled in the data parsed from
    configmanagement.spec.policyController
  Raises: Error, if Policy Controller `enabled` is missing or not a boolean
  """
    if 'spec' not in configmanagement or 'policyController' not in configmanagement['spec']:
        return None
    spec_policy_controller = configmanagement['spec']['policyController']
    if configmanagement['spec']['policyController'] is None or 'enabled' not in spec_policy_controller:
        raise exceptions.Error('Missing required field .spec.policyController.enabled')
    enabled = spec_policy_controller['enabled']
    if not isinstance(enabled, bool):
        raise exceptions.Error('policyController.enabled should be `true` or `false`')
    policy_controller = msg.ConfigManagementPolicyController()
    for field in spec_policy_controller:
        if field not in ['enabled', 'templateLibraryInstalled', 'auditIntervalSeconds', 'referentialRulesEnabled', 'exemptableNamespaces', 'logDeniesEnabled', 'mutationEnabled', 'monitoring']:
            raise exceptions.Error('Please remove illegal field .spec.policyController.{}'.format(field))
        if field == 'monitoring':
            monitoring = _build_monitoring_msg(spec_policy_controller[field], msg)
            setattr(policy_controller, field, monitoring)
        else:
            setattr(policy_controller, field, spec_policy_controller[field])
    return policy_controller