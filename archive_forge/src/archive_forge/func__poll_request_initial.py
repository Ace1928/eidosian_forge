from libcloud.utils.py3 import httplib
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
def _poll_request_initial(self, **kwargs):
    if self.async_request_counter == 0:
        self.poll_response_initial = super().request(**kwargs)
        r = self.poll_response_initial
        self.async_request_counter += 1
    else:
        r = self.request(**kwargs)
    return r