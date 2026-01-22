from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def Justification():
    return base.Argument('--justification', help='A short statement to justify quota increase requests.')