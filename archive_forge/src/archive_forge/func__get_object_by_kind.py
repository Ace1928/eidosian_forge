import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _get_object_by_kind(self, url):
    """
        Fetch a resource and return its object representation by mapping its
        'kind' parameter to the appropriate class.  Returns ``None`` if url is
        ``None``

        :param  url: fully qualified URL of the resource to request from GCE
        :type   url: ``str``

        :return:  Object representation of the requested resource.
        "rtype:   :class:`object` or ``None``
        """
    if not url:
        return None
    response = self.connection.request(url, method='GET').object
    return GCENodeDriver.KIND_METHOD_MAP[response['kind']](self, response)