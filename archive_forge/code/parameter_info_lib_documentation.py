from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.resource import resource_property
Executes command in the current CLI.

    Args:
      command: The command arg list to execute.
      call_arg_complete: Enable arg completion if True.

    Returns:
      Returns the list of resources from the command.
    