from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
class InstancesAddMetadata(base.UpdateCommand):
    """Add or update instance metadata."""

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser, operation_type='set metadata on')
        metadata_utils.AddMetadataArgs(parser, required=True)

    def CreateReference(self, client, resources, args):
        return flags.INSTANCE_ARG.ResolveAsResource(args, resources, scope_lister=flags.GetInstanceZoneScopeLister(client))

    def GetGetRequest(self, client, instance_ref):
        return (client.apitools_client.instances, 'Get', client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))

    def GetSetRequest(self, client, instance_ref, replacement):
        return (client.apitools_client.instances, 'SetMetadata', client.messages.ComputeInstancesSetMetadataRequest(metadata=replacement.metadata, **instance_ref.AsDict()))

    def Modify(self, client, args, existing):
        new_object = encoding.CopyProtoMessage(existing)
        existing_metadata = existing.metadata
        new_object.metadata = metadata_utils.ConstructMetadataMessage(client.messages, metadata=args.metadata, metadata_from_file=args.metadata_from_file, existing_metadata=existing_metadata)
        if metadata_utils.MetadataEqual(existing_metadata, new_object.metadata):
            return None
        else:
            return new_object

    def Run(self, args):
        if not args.metadata and (not args.metadata_from_file):
            raise calliope_exceptions.OneOfArgumentsRequiredException(['--metadata', '--metadata-from-file'], 'At least one of [--metadata] or [--metadata-from-file] must be provided.')
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        project_ref = self.CreateReference(client, holder.resources, args)
        get_request = self.GetGetRequest(client, project_ref)
        objects = client.MakeRequests([get_request])
        new_object = self.Modify(client, args, objects[0])
        if not new_object or objects[0] == new_object:
            log.status.Print('No change requested; skipping update for [{0}].'.format(objects[0].name))
            return objects
        return client.MakeRequests([self.GetSetRequest(client, project_ref, new_object)])