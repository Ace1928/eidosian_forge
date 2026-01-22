import copy
import hmac
import base64
import hashlib
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.compute.types import InvalidCredsError
class CloudStackResponse(JsonResponse):

    def parse_error(self):
        if self.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError('Invalid provider credentials')
        value = None
        body = self.parse_body()
        if hasattr(body, 'values'):
            values = list(body.values())[0]
            if 'errortext' in values:
                value = values['errortext']
        if value is None:
            value = self.body
        if not value:
            value = 'WARNING: error message text sent by provider was empty.'
        error = ProviderError(value=value, http_code=self.status, driver=self.connection.driver)
        raise error