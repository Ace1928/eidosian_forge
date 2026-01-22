from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddListingShareSettingsArgToParser(parser):
    """Add --share-setting flag."""
    parser.add_argument('--share-settings', action='store_true', help='If provided, shows details for the share setting')