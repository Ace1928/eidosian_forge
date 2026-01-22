from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.redis.v1 import redis_v1_messages as messages
def GetCertificateAuthority(self, request, global_params=None):
    """Gets the details of certificate authority information for Redis cluster.

      Args:
        request: (RedisProjectsLocationsClustersGetCertificateAuthorityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CertificateAuthority) The response message.
      """
    config = self.GetMethodConfig('GetCertificateAuthority')
    return self._RunMethod(config, request, global_params=global_params)