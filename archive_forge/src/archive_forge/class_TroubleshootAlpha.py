from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.os_config import troubleshooter
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class TroubleshootAlpha(base.Command):
    """(ALPHA) Troubleshoot VM Manager issues."""

    def _ResolveInstance(self, holder, compute_client, args):
        """Resolves the arguments into an instance.

    Args:
      holder: the api holder
      compute_client: the compute client
      args: The command line arguments.

    Returns:
      An instance reference to a VM.
    """
        resources = holder.resources
        instance_ref = flags.INSTANCE_ARG.ResolveAsResource(args, resources, scope_lister=flags.GetInstanceZoneScopeLister(compute_client))
        return instance_ref

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser)
        parser.add_argument('--enable-log-analysis', required=False, action='store_true', help="Enable the checking of audit logs created by Cloud Logging. The troubleshooter checks the VM's Cloud Logging logs and serial log output for errors, provides you with the analysis data, and allows you to download the logs.")

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        compute_client = holder.client
        instance_ref = self._ResolveInstance(holder, compute_client, args)
        troubleshooter.Troubleshoot(compute_client, instance_ref, self.ReleaseTrack(), analyze_logs=args.enable_log_analysis)
        return