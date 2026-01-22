from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
def StopAirflowCommand(self, request, global_params=None):
    """Stops Airflow CLI command execution.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsStopAirflowCommandRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StopAirflowCommandResponse) The response message.
      """
    config = self.GetMethodConfig('StopAirflowCommand')
    return self._RunMethod(config, request, global_params=global_params)