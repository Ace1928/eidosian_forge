from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def GetFleetProject(args):
    """Gets and parses the fleet project argument.

  Project ID if specified is converted to project number. The parsed fleet
  project has format projects/<project-number>.

  Args:
    args: Arguments parsed from the command.

  Returns:
    The fleet project in format projects/<project-number>
    or None if the fleet projectnot is not specified.
  """
    p = getattr(args, 'fleet_project', None)
    if not p:
        return None
    if not p.isdigit():
        return 'projects/{}'.format(project_util.GetProjectNumber(p))
    return 'projects/{}'.format(p)