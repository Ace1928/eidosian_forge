from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.scc.manage import constants
def CreateParentFlag(required=False) -> base.Argument:
    """Returns a flag for capturing an org, project, or folder name.

  The flag can be provided in one of 2 forms:
    1. --parent=organizations/<id>, --parent=projects/<id or name>,
    --parent=folders/<id>
    2. One of:
      * --organizations=<id> or organizations/<id>
      * --projects=<id or name> or projects/<id or name>
      * --folders=<id> or folders/<id>

  Args:
    required: whether or not this flag is required
  """
    root = base.ArgumentGroup(mutex=True, required=required)
    root.AddArgument(base.Argument('--parent', required=False, help='Parent associated with the custom module. Can be one of\n              organizations/<id>, projects/<id or name>, folders/<id>'))
    root.AddArgument(base.Argument('--organization', required=False, metavar='ORGANIZATION_ID', completer=completers.OrganizationCompleter, help='Organization associated with the custom module.'))
    root.AddArgument(base.Argument('--project', required=False, metavar='PROJECT_ID_OR_NUMBER', completer=completers.ProjectCompleter, help='Project associated with the custom module.'))
    root.AddArgument(base.Argument('--folder', required=False, metavar='FOLDER_ID', help='Folder associated with the custom module.'))
    return root