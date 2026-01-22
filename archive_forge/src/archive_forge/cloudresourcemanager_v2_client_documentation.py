from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v2 import cloudresourcemanager_v2_messages as messages
Cancels the deletion request for a Folder. This method may only be.
called on a Folder in the
DELETE_REQUESTED state.
In order to succeed, the Folder's parent must be in the
ACTIVE state.
In addition, reintroducing the folder into the tree must not violate
folder naming, height and fanout constraints described in the
CreateFolder documentation.
The caller must have `resourcemanager.folders.undelete` permission on the
identified folder.

      Args:
        request: (CloudresourcemanagerFoldersUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Folder) The response message.
      