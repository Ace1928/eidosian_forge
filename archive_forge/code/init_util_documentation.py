from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.resource_manager import operations
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.command_lib.projects import util as  projects_command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
Allows user to select a project.

  Args:
    preselected: str, use this value if not None

  Returns:
    str, project_id or None if was not selected.
  