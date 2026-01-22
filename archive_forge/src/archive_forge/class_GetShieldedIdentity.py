from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags
class GetShieldedIdentity(base.DescribeCommand):
    """Get the Shielded identity for a Compute Engine instance.

  *{command}* displays the Shielded identity associated with a
  Compute Engine instance in a project.
  """
    detailed_help = {'EXAMPLES': '\n  To get the shielded identity for an instance, run:\n\n    $ {command} example-instance --zone=us-central1-b\n  '}

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser, operation_type='describe the Shielded identity of')
        base.URI_FLAG.RemoveFromParser(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
        request = (client.apitools_client.instances, 'GetShieldedInstanceIdentity', client.messages.ComputeInstancesGetShieldedInstanceIdentityRequest(instance=instance_ref.instance, zone=instance_ref.zone, project=instance_ref.project))
        errors = []
        objects = client.MakeRequests(requests=[request], errors_to_collect=errors)
        if errors:
            utils.RaiseToolException(errors, error_message='Could not get Shielded identity:')
        response = objects[0]
        return response