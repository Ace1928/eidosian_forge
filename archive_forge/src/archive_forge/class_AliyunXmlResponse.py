import sys
import hmac
import time
import uuid
import base64
import hashlib
from libcloud.utils.py3 import ET, b, u, urlquote
from libcloud.utils.xml import findtext
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import MalformedResponseError
class AliyunXmlResponse(XmlResponse):
    namespace = None

    def success(self):
        return 200 <= self.status < 300

    def parse_body(self):
        """
        Each response from Aliyun contains a request id and a host id.
        The response body is in utf-8 encoding.
        """
        if len(self.body) == 0 and (not self.parse_zero_length_body):
            return self.body
        try:
            parser = ET.XMLParser(encoding='utf-8')
            body = ET.XML(self.body.encode('utf-8'), parser=parser)
        except Exception:
            raise MalformedResponseError('Failed to parse XML', body=self.body, driver=self.connection.driver)
        self.request_id = findtext(element=body, xpath='RequestId', namespace=self.namespace)
        self.host_id = findtext(element=body, xpath='HostId', namespace=self.namespace)
        return body

    def parse_error(self):
        """
        Parse error responses from Aliyun.
        """
        body = super().parse_error()
        code, message = self._parse_error_details(element=body)
        request_id = findtext(element=body, xpath='RequestId', namespace=self.namespace)
        host_id = findtext(element=body, xpath='HostId', namespace=self.namespace)
        error = {'code': code, 'message': message, 'request_id': request_id, 'host_id': host_id}
        return u(error)

    def _parse_error_details(self, element):
        """
        Parse error code and message from the provided error element.

        :return: ``tuple`` with two elements: (code, message)
        :rtype: ``tuple``
        """
        code = findtext(element=element, xpath='Code', namespace=self.namespace)
        message = findtext(element=element, xpath='Message', namespace=self.namespace)
        return (code, message)