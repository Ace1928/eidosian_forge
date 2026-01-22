from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
def PollAirflowCommand(self, request, global_params=None):
    """Polls Airflow CLI command execution and fetches logs.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsPollAirflowCommandRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PollAirflowCommandResponse) The response message.
      """
    config = self.GetMethodConfig('PollAirflowCommand')
    return self._RunMethod(config, request, global_params=global_params)