import hmac
import time
import hashlib
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Container, StorageDriver
class NimbusResponse(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.NOT_FOUND, httplib.CONFLICT, httplib.BAD_REQUEST]

    def success(self):
        return self.status in self.valid_response_codes

    def parse_error(self):
        if self.status in [httplib.UNAUTHORIZED]:
            raise InvalidCredsError(self.body)
        raise LibcloudError('Unknown error. Status code: %d' % self.status, driver=self.connection.driver)