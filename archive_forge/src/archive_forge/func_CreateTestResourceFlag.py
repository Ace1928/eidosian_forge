from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.scc.manage import constants
def CreateTestResourceFlag(required=True) -> base.Argument:
    return base.Argument('--resource-from-file', required=required, metavar='TEST_DATA', help='Path to a YAML file that contains the resource data to validate the Security Health Analytics custom module against.', type=arg_parsers.FileContents())