from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.scc.manage import constants
def CreateCustomConfigFlag(required=True) -> base.Argument:
    return base.Argument('--custom-config-from-file', required=required, metavar='CUSTOM_CONFIG', help='Path to a YAML custom configuration file.', type=arg_parsers.FileContents())