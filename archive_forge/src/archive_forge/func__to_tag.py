import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_tag(self, element):
    tag_key = self._to_tag_key(element, from_tag_api=True)
    return NttCisTag(asset_type=findtext(element, 'assetType', TYPES_URN), asset_id=findtext(element, 'assetId', TYPES_URN), asset_name=findtext(element, 'assetName', TYPES_URN), datacenter=findtext(element, 'datacenterId', TYPES_URN), key=tag_key, value=findtext(element, 'value', TYPES_URN))