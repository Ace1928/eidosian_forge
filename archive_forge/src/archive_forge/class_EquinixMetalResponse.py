import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
class EquinixMetalResponse(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.ACCEPTED, httplib.CREATED, httplib.NO_CONTENT]

    def parse_error(self):
        if self.status == httplib.UNAUTHORIZED:
            body = self.parse_body()
            raise InvalidCredsError(body.get('error'))
        else:
            body = self.parse_body()
            if 'message' in body:
                error = '{} (code: {})'.format(body.get('message'), self.status)
            elif 'errors' in body:
                error = body.get('errors')
            else:
                error = body
            raise Exception(error)

    def success(self):
        return self.status in self.valid_response_codes