from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1alpha1 import cloudbuild_v1alpha1_messages as messages
Update a `WorkerPool`.

      Args:
        request: (CloudbuildProjectsWorkerPoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkerPool) The response message.
      