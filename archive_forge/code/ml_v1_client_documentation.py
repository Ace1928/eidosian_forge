from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.ml.v1 import ml_v1_messages as messages
Performs online prediction on the data in the request. {% dynamic include "/ai-platform/includes/___predict-request" %} .

      Args:
        request: (MlProjectsPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      