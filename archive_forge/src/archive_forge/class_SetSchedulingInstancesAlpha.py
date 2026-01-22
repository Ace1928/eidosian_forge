from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import flags as sole_tenancy_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core.util import times
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SetSchedulingInstancesAlpha(SetSchedulingInstancesBeta):
    """Set scheduling options for Compute Engine virtual machines.

    *${command}* is used to update scheduling options for VM instances.
    You can only call this method on a VM instance that is stopped
    (a VM instance in a `TERMINATED` state).
  """
    _support_host_error_timeout_seconds = True
    _support_local_ssd_recovery_timeout = True
    _support_max_run_duration = True
    _support_graceful_shutdown = True

    @classmethod
    def Args(cls, parser):
        parser.add_argument('--restart-on-failure', action=arg_parsers.StoreTrueFalseAction, help='        The instances will be restarted if they are terminated by Compute\n        Engine.  This does not affect terminations performed by the user.\n        This option is mutually exclusive with --preemptible.\n        ')
        flags.AddPreemptibleVmArgs(parser, is_update=True)
        flags.AddProvisioningModelVmArgs(parser)
        flags.AddInstanceTerminationActionVmArgs(parser, is_update=True)
        flags.AddMaintenancePolicyArgs(parser, deprecate=True)
        sole_tenancy_flags.AddNodeAffinityFlagToParser(parser, is_update=True)
        flags.INSTANCE_ARG.AddArgument(parser)
        flags.AddMinNodeCpuArg(parser, is_update=True)
        flags.AddHostErrorTimeoutSecondsArgs(parser)
        flags.AddLocalSsdRecoveryTimeoutArgs(parser)
        flags.AddMaxRunDurationVmArgs(parser, is_update=True)
        flags.AddDiscardLocalSsdVmArgs(parser, is_update=True)
        flags.AddGracefulShutdownArgs(parser)