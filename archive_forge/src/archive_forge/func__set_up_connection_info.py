from libcloud.utils.py3 import ET, httplib
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.types import LibcloudError, MalformedResponseError, KeyPairDoesNotExistError
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.openstack_identity import (
def _set_up_connection_info(self, url):
    prev_conn = (self.host, self.port, self.secure)
    result = self._tuple_from_url(url)
    self.host, self.port, self.secure, self.request_path = result
    new_conn = (self.host, self.port, self.secure)
    if new_conn != prev_conn:
        self.connect()