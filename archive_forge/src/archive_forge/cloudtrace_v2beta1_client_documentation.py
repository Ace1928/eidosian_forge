from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudtrace.v2beta1 import cloudtrace_v2beta1_messages as messages
Updates a sink. This method updates fields in the existing sink according to the provided update mask. The sink's name cannot be changed nor any output-only fields (e.g. the writer_identity).

      Args:
        request: (CloudtraceProjectsTraceSinksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TraceSink) The response message.
      