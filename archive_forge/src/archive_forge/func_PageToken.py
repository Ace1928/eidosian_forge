from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def PageToken():
    return base.Argument('--page-token', default=None, help='A token identifying a page of results the server should return. Default is none.')