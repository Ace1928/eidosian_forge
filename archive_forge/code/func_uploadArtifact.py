from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import transfer
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import scaled_integer
def uploadArtifact(self, args, file_path, client, messages):
    chunksize = scaled_integer.ParseInteger(properties.VALUES.storage.upload_chunk_size.Get())
    repo_ref = args.CONCEPTS.repository.Parse()
    if args.source:
        file_name = os.path.basename(file_path)
        if args.destination_path:
            path = os.path.normpath(args.destination_path)
            file_name = os.path.join(path, os.path.basename(file_path))
    else:
        file_name = file_path[len(args.source_directory) + 1:]
        if args.destination_path:
            path = os.path.normpath(args.destination_path)
            file_name = os.path.join(path, file_name)
    file_name = file_name.replace(os.sep, '/')
    request = messages.ArtifactregistryProjectsLocationsRepositoriesGenericArtifactsUploadRequest(uploadGenericArtifactRequest=messages.UploadGenericArtifactRequest(packageId=args.package, versionId=args.version, filename=file_name), parent=repo_ref.RelativeName())
    mime_type = util.GetMimetype(file_path)
    upload = transfer.Upload.FromFile(file_path, mime_type=mime_type, chunksize=chunksize)
    op_obj = client.projects_locations_repositories_genericArtifacts.Upload(request, upload=upload)
    op = op_obj.operation
    op_ref = resources.REGISTRY.ParseRelativeName(op.name, collection='artifactregistry.projects.locations.operations')
    if args.async_:
        return op_ref
    else:
        result = waiter.WaitFor(waiter.CloudOperationPollerNoResources(client.projects_locations_operations), op_ref, 'Uploading file: {}'.format(file_name))
        return result