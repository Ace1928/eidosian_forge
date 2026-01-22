from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddNoRestartOnFailureArgs(parser):
    parser.add_argument('--restart-on-failure', action='store_true', default=True, help='      The VMs created from the imported machine image are restarted if\n      they are terminated by Compute Engine. This does not affect terminations\n      performed by the user.\n      ')