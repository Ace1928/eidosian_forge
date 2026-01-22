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
def _files_to_xml(self, files):
    if not files:
        return None
    personality_elm = ET.Element('personality')
    for k, v in list(files.items()):
        file_elm = ET.SubElement(personality_elm, 'file', {'path': str(k)})
        file_elm.text = base64.b64encode(b(v)).decode('ascii')
    return personality_elm