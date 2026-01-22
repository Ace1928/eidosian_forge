from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
def SendVerificationCode(self, request, global_params=None):
    """Causes a verification code to be delivered to the channel. The code can then be supplied in VerifyNotificationChannel to verify the channel.

      Args:
        request: (MonitoringProjectsNotificationChannelsSendVerificationCodeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('SendVerificationCode')
    return self._RunMethod(config, request, global_params=global_params)