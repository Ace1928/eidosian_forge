from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.scc.manage import constants
def CreateEtdCustomConfigFilePathFlag(required=True) -> base.Argument:
    return base.Argument('--custom-config-file', required=required, metavar='CUSTOM_CONFIG', help='Path to a JSON custom configuration file of the ETD custom module.', type=arg_parsers.FileContents())