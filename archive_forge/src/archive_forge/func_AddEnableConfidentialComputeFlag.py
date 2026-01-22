from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def AddEnableConfidentialComputeFlag(parser):
    return parser.add_argument('--confidential-compute', action='store_true', help='\n      Creates the disk with confidential compute mode enabled. Encryption with a Cloud KMS key is required to enable this option.\n      ')