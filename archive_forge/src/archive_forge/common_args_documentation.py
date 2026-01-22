from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers as resource_manager_completers
from googlecloudsdk.core import properties
Creates project argument.

  Args:
    help_text_to_prepend: str, help text to prepend to the generic --project
      help text.
    help_text_to_overwrite: str, help text to overwrite the generic --project
      help text.

  Returns:
    calliope.base.Argument, The argument for project.
  