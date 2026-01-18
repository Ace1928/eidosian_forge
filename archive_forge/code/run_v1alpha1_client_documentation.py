from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1alpha1 import run_v1alpha1_messages as messages
Rpc to replace a domain mapping. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (RunProjectsLocationsDomainmappingsReplaceDomainMappingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DomainMapping) The response message.
      