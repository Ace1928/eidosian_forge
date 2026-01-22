from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddAdminEnabledForUpdate(parser):
    """Adds adminEnabled flag to the argparse.ArgumentParser."""
    admin_enabled_args = parser.add_mutually_exclusive_group()
    admin_enabled_args.add_argument('--admin-enabled', action='store_true', default=None, help='      Administrative status of the interconnect.\n      When this is enabled, the interconnect is operational and will carry\n      traffic across any functioning linked interconnect attachments. Use\n      --no-admin-enabled to disable it.\n      ')