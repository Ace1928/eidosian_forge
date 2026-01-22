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
def _get_volume_connection(self):
    """
        Get the correct Volume connection (v3 or v2)
        """
    if not self.volume_connection:
        try:
            self.volumev3_connection.get_service_catalog()
            self.volume_connection = self.volumev3_connection
        except LibcloudError:
            self.volume_connection = self.volumev2_connection
    return self.volume_connection