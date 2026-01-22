from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import automation
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import resources
Deletes an automation resource by calling the delete automation API.

  Args:
    name: str, automation name.

  Returns:
    The operation message.
  