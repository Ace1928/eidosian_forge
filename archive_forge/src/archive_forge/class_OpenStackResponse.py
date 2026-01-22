from libcloud.utils.py3 import ET, httplib
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.types import LibcloudError, MalformedResponseError, KeyPairDoesNotExistError
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.openstack_identity import (
class OpenStackResponse(Response):
    node_driver = None

    def success(self):
        i = int(self.status)
        return 200 <= i <= 299

    def has_content_type(self, content_type):
        content_type_value = self.headers.get('content-type') or ''
        content_type_value = content_type_value.lower()
        return content_type_value.find(content_type.lower()) > -1

    def parse_body(self):
        if self.status == httplib.NO_CONTENT or not self.body:
            return None
        if self.has_content_type('application/xml'):
            try:
                return ET.XML(self.body)
            except Exception:
                raise MalformedResponseError('Failed to parse XML', body=self.body, driver=self.node_driver)
        elif self.has_content_type('application/json'):
            try:
                return json.loads(self.body)
            except Exception:
                raise MalformedResponseError('Failed to parse JSON', body=self.body, driver=self.node_driver)
        else:
            return self.body

    def parse_error(self):
        body = self.parse_body()
        if self.has_content_type('application/xml'):
            text = '; '.join([err.text or '' for err in body.getiterator() if err.text])
        elif self.has_content_type('application/json'):
            values = list(body.values())
            context = self.connection.context
            driver = self.connection.driver
            key_pair_name = context.get('key_pair_name', None)
            if len(values) > 0 and 'code' in values[0] and (values[0]['code'] == 404) and key_pair_name:
                raise KeyPairDoesNotExistError(name=key_pair_name, driver=driver)
            elif len(values) > 0 and 'message' in values[0]:
                text = ';'.join([fault_data['message'] for fault_data in values])
            else:
                text = body
        else:
            text = body
        return '{} {} {}'.format(self.status, self.error, text)