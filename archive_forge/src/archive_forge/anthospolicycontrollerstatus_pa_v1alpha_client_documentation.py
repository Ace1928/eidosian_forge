from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthospolicycontrollerstatus_pa.v1alpha import anthospolicycontrollerstatus_pa_v1alpha_messages as messages
ListMembershipsProducer returns runtime status from memberships of a fleet.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsMembershipsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipsProducerResponse) The response message.
      