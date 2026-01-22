from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class InstanceSetName(base.SilentCommand):
    """Set name for Compute Engine virtual machine instances."""

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser)
        parser.add_argument('--new-name', required=True, help='        Specifies the new name of the instance. ')

    def _CreateSetNameRequest(self, client, instance_ref, name):
        return (client.apitools_client.instances, 'SetName', client.messages.ComputeInstancesSetNameRequest(instancesSetNameRequest=client.messages.InstancesSetNameRequest(name=name, currentName=instance_ref.Name()), **instance_ref.AsDict()))

    def _CreateGetRequest(self, client, instance_ref):
        return (client.apitools_client.instances, 'Get', client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=flags.GetInstanceZoneScopeLister(client))
        get_request = self._CreateGetRequest(client, instance_ref)
        objects = client.MakeRequests([get_request])
        if args.new_name == objects[0].name:
            return objects[0]
        set_request = self._CreateSetNameRequest(client, instance_ref, args.new_name)
        return client.MakeRequests([set_request], followup_overrides=[args.new_name])