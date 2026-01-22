from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
def Trigger(self, request, global_params=None):
    """Trigger a DAG run.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsTriggerRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DagRun) The response message.
      """
    config = self.GetMethodConfig('Trigger')
    return self._RunMethod(config, request, global_params=global_params)