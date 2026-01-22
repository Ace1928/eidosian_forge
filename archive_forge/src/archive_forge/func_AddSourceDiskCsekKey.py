from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddSourceDiskCsekKey(parser):
    parser.add_argument('--source-disk-key-file', metavar='FILE', help='\n      Path to the customer-supplied encryption key of the source disk.\n      Required if the source disk is protected by a customer-supplied\n      encryption key.\n      ')