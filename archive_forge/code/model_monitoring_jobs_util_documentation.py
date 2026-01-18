from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
Parse operation relative resource name to the operation reference object.

  Args:
    operation_name: The operation resource name

  Returns:
    The operation reference object
  