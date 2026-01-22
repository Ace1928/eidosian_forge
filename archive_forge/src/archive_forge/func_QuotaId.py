from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def QuotaId(positional=True, text='ID of the quota, which is unique within the service.'):
    if positional:
        return base.Argument('QUOTA_ID', type=str, help=text)
    else:
        return base.Argument('--quota-id', type=str, required=True, help=text)