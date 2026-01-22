from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
Lists TrainingPipelines in a Location.

      Args:
        request: (AiplatformProjectsLocationsTrainingPipelinesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListTrainingPipelinesResponse) The response message.
      