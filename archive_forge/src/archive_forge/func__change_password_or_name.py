import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _change_password_or_name(self, node, name=None, password=None):
    uri = '/servers/%s' % node.id
    if not name:
        name = node.name
    body = {'xmlns': self.XML_NAMESPACE, 'name': name}
    if password is not None:
        body['adminPass'] = password
    server_elm = ET.Element('server', body)
    resp = self.connection.request(uri, method='PUT', data=ET.tostring(server_elm))
    if resp.status == httplib.NO_CONTENT and password is not None:
        node.extra['password'] = password
    return resp.status == httplib.NO_CONTENT