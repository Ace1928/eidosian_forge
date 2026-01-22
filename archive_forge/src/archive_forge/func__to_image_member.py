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
def _to_image_member(self, api_image_member):
    created = api_image_member['created_at']
    updated = api_image_member.get('updated_at')
    return NodeImageMember(id=api_image_member['member_id'], image_id=api_image_member['image_id'], state=api_image_member['status'], created=created, driver=self, extra=dict(schema=api_image_member.get('schema'), updated=updated))