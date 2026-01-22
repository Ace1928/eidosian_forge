import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
class CloudSigma_2_0_Response(JsonResponse):
    success_status_codes = [httplib.OK, httplib.ACCEPTED, httplib.NO_CONTENT, httplib.CREATED]

    def success(self):
        return self.status in self.success_status_codes

    def parse_error(self):
        if int(self.status) == httplib.UNAUTHORIZED:
            raise InvalidCredsError('Invalid credentials')
        body = self.parse_body()
        errors = self._parse_errors_from_body(body=body)
        if errors:
            raise errors[0]
        return body

    def _parse_errors_from_body(self, body):
        """
        Parse errors from the response body.

        :return: List of error objects.
        :rtype: ``list`` of :class:`.CloudSigmaError` objects
        """
        errors = []
        if not isinstance(body, list):
            return None
        for item in body:
            if 'error_type' not in item:
                continue
            error = CloudSigmaError(http_code=self.status, error_type=item['error_type'], error_msg=item['error_message'], error_point=item['error_point'], driver=self.connection.driver)
            errors.append(error)
        return errors