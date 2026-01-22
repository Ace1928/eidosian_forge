from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def SetExecutionConfig(messages, target, target_ref, execution_configs):
    """Sets the executionConfigs field of cloud deploy resource message.

  Args:
    messages: module containing the definitions of messages for Cloud Deploy.
    target:  googlecloudsdk.generated_clients.apis.clouddeploy.Target message.
    target_ref: protorpc.messages.Message, target resource object.
    execution_configs:
      [googlecloudsdk.generated_clients.apis.clouddeploy.ExecutionConfig], list
      of ExecutionConfig messages.

  Raises:
    arg_parsers.ArgumentTypeError: if usage is not a valid enum.
  """
    _EnsureIsType(execution_configs, list, 'failed to parse target {}, executionConfigs are defined incorrectly'.format(target_ref.Name()))
    for config in execution_configs:
        execution_config_message = messages.ExecutionConfig()
        for field in config:
            if field != 'usages':
                setattr(execution_config_message, field, config.get(field))
        usages = config.get('usages') or []
        for usage in usages:
            execution_config_message.usages.append(arg_utils.ChoiceToEnum(usage, messages.ExecutionConfig.UsagesValueListEntryValuesEnum, valid_choices=USAGE_CHOICES))
        target.executionConfigs.append(execution_config_message)