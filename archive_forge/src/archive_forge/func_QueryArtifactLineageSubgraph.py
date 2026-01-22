from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def QueryArtifactLineageSubgraph(self, request, global_params=None):
    """Retrieves lineage of an Artifact represented through Artifacts and Executions connected by Event edges and returned as a LineageSubgraph.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresArtifactsQueryArtifactLineageSubgraphRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1LineageSubgraph) The response message.
      """
    config = self.GetMethodConfig('QueryArtifactLineageSubgraph')
    return self._RunMethod(config, request, global_params=global_params)