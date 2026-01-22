import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _image_needs_auth(self, image):
    if not isinstance(image, NodeImage):
        image = self.ex_get_image_by_id(image)
    if image.extra['isCustomerImage'] and image.extra['OS_type'] == 'UNIX':
        return False
    return True