from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def ParseReplacementMethod(method_type_str, messages):
    """Retrieves value of update policy type: substitute or recreate.

  Args:
    method_type_str: string containing update policy type.
    messages: module containing message classes.

  Returns:
    InstanceGroupManagerUpdatePolicy.TypeValueValuesEnum message enum value.
  """
    return arg_utils.ChoiceToEnum(method_type_str, messages.InstanceGroupManagerUpdatePolicy.ReplacementMethodValueValuesEnum)