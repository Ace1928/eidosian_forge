from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1alpha1 import anthosevents_v1alpha1_messages as messages
Rpc to replace a CloudRun resource. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesCloudrunsReplaceCloudRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudRun) The response message.
      