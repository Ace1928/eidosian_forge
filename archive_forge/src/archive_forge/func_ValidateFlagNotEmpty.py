from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import exceptions
def ValidateFlagNotEmpty(value, flag_name):
    """Ensures that value is not empty.

  Args:
    value: str, the value to check
    flag_name: str, the flag's name; included in the exception's message

  Raises:
    exceptions.Error: if value is empty
  """
    if not value:
        raise exceptions.Error('Missing required flag ' + flag_name)