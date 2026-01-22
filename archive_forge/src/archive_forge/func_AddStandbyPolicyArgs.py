from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.health_checks import flags as health_checks_flags
def AddStandbyPolicyArgs(standby_policy_params):
    """Adds autohealing-related commandline arguments to parser."""
    standby_policy_params.add_argument('--initial-delay', type=_InitialDelayValidator, help='      Initialization delay before stopping or suspending instances\n      in this managed instance group. For example: 5m or 300s.\n      See `gcloud topic datetimes` for information on duration formats.\n      ')