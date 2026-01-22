from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
def ListChangedFiles(self, request, global_params=None):
    """ListChangedFiles computes the files that have changed between two revisions.
or workspace snapshots in the same repo. It returns a list of
ChangeFileInfos.

ListChangedFiles does not perform copy/rename detection, so the from_path of
ChangeFileInfo is unset. Examine the changed_files field of the Revision
resource to determine copy/rename information.

The result is ordered by path. Pagination is supported.

      Args:
        request: (ListChangedFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListChangedFilesResponse) The response message.
      """
    config = self.GetMethodConfig('ListChangedFiles')
    return self._RunMethod(config, request, global_params=global_params)