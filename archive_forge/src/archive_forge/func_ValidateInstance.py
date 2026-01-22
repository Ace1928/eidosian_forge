from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import exceptions
def ValidateInstance(value, param_name):
    """Performs syntax check on an instance value; doesn't check whether it exists.

  Args:
    value: str, the instance value to check
    param_name: str, the parameter's name; included in the exception's message

  Raises:
    exceptions.Error: if value is empty
  """
    if not value:
        raise exceptions.Error('Missing required parameter ' + param_name)