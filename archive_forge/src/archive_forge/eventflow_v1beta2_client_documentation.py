from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.eventflow.v1beta2 import eventflow_v1beta2_messages as messages
Updates a flow, returning the updated flow. Empty fields (proto3 default values) mean don't change those fields. The call returns INVALID_ARGUMENT status if the spec.name, spec.namespace, or spec.trigger.event_type is change. trigger.event_type is changed.

      Args:
        request: (EventflowProjectsFlowsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Flow) The response message.
      