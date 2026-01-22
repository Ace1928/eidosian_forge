from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouderrorreporting.v1beta1 import clouderrorreporting_v1beta1_messages as messages
Deletes all error events of a given project.

      Args:
        request: (ClouderrorreportingProjectsDeleteEventsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeleteEventsResponse) The response message.
      