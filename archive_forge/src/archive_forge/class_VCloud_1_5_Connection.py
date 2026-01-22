import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class VCloud_1_5_Connection(VCloudConnection):

    def _get_auth_headers(self):
        """Compatibility for using v1.5 API under vCloud Director 5.1"""
        return {'Authorization': 'Basic %s' % base64.b64encode(b('{}:{}'.format(self.user_id, self.key))).decode('utf-8'), 'Content-Length': '0', 'Accept': 'application/*+xml;version=1.5'}

    def _get_auth_token(self):
        if not self.token:
            self.connection.request(method='POST', url='/api/sessions', headers=self._get_auth_headers())
            resp = self.connection.getresponse()
            headers = resp.headers
            try:
                self.token = headers['x-vcloud-authorization']
            except KeyError:
                raise InvalidCredsError()
            body = ET.XML(resp.text)
            self.org_name = body.get('org')
            org_list_url = get_url_path(next((link for link in body.findall(fixxpath(body, 'Link')) if link.get('type') == 'application/vnd.vmware.vcloud.orgList+xml')).get('href'))
            if self.proxy_url is not None:
                self.connection.set_http_proxy(self.proxy_url)
            self.connection.request(method='GET', url=org_list_url, headers=self.add_default_headers({}))
            body = ET.XML(self.connection.getresponse().text)
            self.driver.org = get_url_path(next((org for org in body.findall(fixxpath(body, 'Org')) if org.get('name') == self.org_name)).get('href'))

    def add_default_headers(self, headers):
        headers['Accept'] = 'application/*+xml;version=1.5'
        headers['x-vcloud-authorization'] = self.token
        return headers