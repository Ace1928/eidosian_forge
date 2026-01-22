from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddMaxRetentionDays(parser):
    parser.add_argument('--max-retention-days', help='\n    Days for snapshot to live before being automatically deleted. If unspecified, the snapshot will live until manually deleted.\n    ')