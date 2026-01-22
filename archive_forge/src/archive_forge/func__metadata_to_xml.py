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
def _metadata_to_xml(self, metadata):
    if not metadata:
        return None
    metadata_elm = ET.Element('metadata')
    for k, v in list(metadata.items()):
        meta_elm = ET.SubElement(metadata_elm, 'meta', {'key': str(k)})
        meta_elm.text = str(v)
    return metadata_elm