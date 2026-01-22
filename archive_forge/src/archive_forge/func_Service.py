from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def Service():
    return base.Argument('--service', required=True, help='Name of the service in which the quota is defined.')