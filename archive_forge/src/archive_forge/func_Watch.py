from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.runtimeconfig.v1beta1 import runtimeconfig_v1beta1_messages as messages
def Watch(self, request, global_params=None):
    """Watches a specific variable and waits for a change in the variable's value. When there is a change, this method returns the new value or times out. If a variable is deleted while being watched, the `variableState` state is set to `DELETED` and the method returns the last known variable `value`. If you set the deadline for watching to a larger value than internal timeout (60 seconds), the current variable value is returned and the `variableState` will be `VARIABLE_STATE_UNSPECIFIED`. To learn more about creating a watcher, read the [Watching a Variable for Changes](/deployment-manager/runtime-configurator/watching-a-variable) documentation.

      Args:
        request: (RuntimeconfigProjectsConfigsVariablesWatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Variable) The response message.
      """
    config = self.GetMethodConfig('Watch')
    return self._RunMethod(config, request, global_params=global_params)