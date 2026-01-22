from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v2alpha1 import cloudresourcemanager_v2alpha1_messages as messages
Updates a Folder, changing its display_name. Changes to the folder display_name will be rejected if they violate either the display_name formatting rules or naming constraints described in the [CreateFolder] documentation. + The Folder's display name must start and end with a letter or digit, may contain letters, digits, spaces, hyphens and underscores and can be no longer than 30 characters. This is captured by the regular expression: `[\p{L}\p{N}]({\p{L}\p{N}_- ]{0,28}[\p{L}\p{N}])?`. The caller must have `resourcemanager.folders.update` permission on the identified folder.

      Args:
        request: (CloudresourcemanagerFoldersUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Folder) The response message.
      