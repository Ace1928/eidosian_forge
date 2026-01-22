from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import time
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves import range  # pylint: disable=redefined-builtin
def _PollUntilDone(self, operation, retry_callback):
    """Polls the operation resource until it is complete or times out."""
    if operation.done:
        return operation
    request_type = self.client.operations.GetRequestType('Get')
    request = request_type(name=operation.name)
    for _ in range(self._MAX_RETRIES):
        operation = self.client.operations.Get(request)
        if operation.done:
            log.debug('Operation [{0}] complete. Result: {1}'.format(operation.name, json.dumps(encoding.MessageToDict(operation), indent=4)))
            return operation
        log.debug('Operation [{0}] not complete. Waiting {1}s.'.format(operation.name, self._RETRY_INTERVAL))
        time.sleep(self._RETRY_INTERVAL)
        if retry_callback is not None:
            retry_callback()
    return None