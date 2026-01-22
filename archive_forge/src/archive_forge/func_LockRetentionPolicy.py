import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
def LockRetentionPolicy(self, request, global_params=None):
    """Locks retention policy on a bucket.

      Args:
        request: (StorageBucketsLockRetentionPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Bucket) The response message.
      """
    config = self.GetMethodConfig('LockRetentionPolicy')
    return self._RunMethod(config, request, global_params=global_params)