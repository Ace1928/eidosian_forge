from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.command_lib.filestore import update_util
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetExistingBackup(unused_resource_ref, args, patch_request):
    """Fetch existing Filestore instance to update and add it to Patch request."""
    resource_ref = GetResourceRef(args)
    api_version = util.GetApiVersionFromArgs(args)
    client = filestore_client.FilestoreClient(api_version)
    orig_backup = client.GetBackup(resource_ref)
    patch_request.backup = orig_backup
    return patch_request