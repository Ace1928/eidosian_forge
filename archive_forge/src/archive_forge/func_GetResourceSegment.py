from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def GetResourceSegment(args):
    """Returns the resource from up to the cloud entity."""
    if hasattr(args, 'project') and args.project:
        return 'projects/%s' % args.project
    elif hasattr(args, 'folder') and args.folder:
        return 'folders/%s' % args.folder
    elif hasattr(args, 'billing_account') and args.billing_account:
        return 'billingAccounts/%s' % args.billing_account
    else:
        return 'organizations/%s' % args.organization