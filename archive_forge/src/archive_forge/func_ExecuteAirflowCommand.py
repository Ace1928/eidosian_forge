from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
def ExecuteAirflowCommand(self, request, global_params=None):
    """Executes Airflow CLI command.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsExecuteAirflowCommandRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExecuteAirflowCommandResponse) The response message.
      """
    config = self.GetMethodConfig('ExecuteAirflowCommand')
    return self._RunMethod(config, request, global_params=global_params)