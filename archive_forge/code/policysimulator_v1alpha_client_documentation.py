from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policysimulator.v1alpha import policysimulator_v1alpha_messages as messages
Lists each Replay in a project, folder, or organization. Each `Replay` is available for at least 7 days.

      Args:
        request: (PolicysimulatorProjectsLocationsReplaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicysimulatorV1alphaListReplaysResponse) The response message.
      