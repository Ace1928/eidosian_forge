from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.secretmanager.v1 import secretmanager_v1_messages as messages
def Access(self, request, global_params=None):
    """Accesses a SecretVersion. This call returns the secret data. `projects/*/secrets/*/versions/latest` is an alias to the most recently created SecretVersion.

      Args:
        request: (SecretmanagerProjectsSecretsVersionsAccessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSecretVersionResponse) The response message.
      """
    config = self.GetMethodConfig('Access')
    return self._RunMethod(config, request, global_params=global_params)