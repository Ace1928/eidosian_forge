from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.scc.manage import constants
def CreateUpdateFlags(module_type: constants.CustomModuleType, file_type, required=True) -> base.Argument:
    """Returns a custom-config flag or an enablement-state flag, or both."""
    root = base.ArgumentGroup(mutex=False, required=required)
    root.AddArgument(base.Argument('--custom-config-file', required=False, default=None, help='Path to a {} file that contains the custom config to set for the module.'.format(file_type), type=arg_parsers.FileContents()))
    root.AddArgument(CreateEnablementStateFlag(required=False, module_type=module_type))
    return root