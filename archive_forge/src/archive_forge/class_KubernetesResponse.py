import os
import base64
import warnings
from typing import Optional
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import (
from libcloud.common.types import InvalidCredsError
class KubernetesResponse(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.ACCEPTED, httplib.CREATED, httplib.NO_CONTENT]

    def parse_error(self):
        if self.status == 401:
            raise InvalidCredsError('Invalid credentials')
        return self.body

    def success(self):
        return self.status in self.valid_response_codes