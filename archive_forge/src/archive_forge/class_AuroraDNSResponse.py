import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
class AuroraDNSResponse(JsonResponse):

    def success(self):
        return self.status in [httplib.OK, httplib.CREATED, httplib.ACCEPTED]

    def parse_error(self):
        status = int(self.status)
        error = {'driver': self, 'value': ''}
        if status == httplib.UNAUTHORIZED:
            error['value'] = 'Authentication failed'
            raise InvalidCredsError(**error)
        elif status == httplib.FORBIDDEN:
            error['value'] = 'Authorization failed'
            error['http_code'] = status
            raise ProviderError(**error)
        elif status == httplib.NOT_FOUND:
            context = self.connection.context
            if context['resource'] == 'zone':
                error['zone_id'] = context['id']
                raise ZoneDoesNotExistError(**error)
            elif context['resource'] == 'record':
                error['record_id'] = context['id']
                raise RecordDoesNotExistError(**error)
            elif context['resource'] == 'healthcheck':
                error['health_check_id'] = context['id']
                raise HealthCheckDoesNotExistError(**error)
        elif status == httplib.CONFLICT:
            context = self.connection.context
            if context['resource'] == 'zone':
                error['zone_id'] = context['id']
                raise ZoneAlreadyExistsError(**error)
        elif status == httplib.BAD_REQUEST:
            context = self.connection.context
            body = self.parse_body()
            raise ProviderError(value=body['errormsg'], http_code=status, driver=self)