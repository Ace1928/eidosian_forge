import copy
import base64
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import ET, b, httplib
from libcloud.utils.xml import findall, findtext
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
class ZerigoDNSResponse(XmlResponse):

    def success(self):
        return self.status in [httplib.OK, httplib.CREATED, httplib.ACCEPTED]

    def parse_error(self):
        status = int(self.status)
        if status == 401:
            if not self.body:
                raise InvalidCredsError(str(self.status) + ': ' + self.error)
            else:
                raise InvalidCredsError(self.body)
        elif status == 404:
            context = self.connection.context
            if context['resource'] == 'zone':
                raise ZoneDoesNotExistError(value='', driver=self, zone_id=context['id'])
            elif context['resource'] == 'record':
                raise RecordDoesNotExistError(value='', driver=self, record_id=context['id'])
        elif status != 503:
            try:
                body = ET.XML(self.body)
            except Exception:
                raise MalformedResponseError('Failed to parse XML', body=self.body)
            errors = []
            for error in findall(element=body, xpath='error'):
                errors.append(error.text)
            raise ZerigoError(code=status, errors=errors)
        return self.body