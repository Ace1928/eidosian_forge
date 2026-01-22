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
def _try_image_family(image_family, project=None):
    request = '/global/images/family/%s' % image_family
    save_request_path = self.connection.request_path
    if project:
        new_request_path = save_request_path.replace(self.project, project)
        self.connection.request_path = new_request_path
    try:
        response = self.connection.request(request, method='GET')
        image = self._to_node_image(response.object)
    except ResourceNotFoundError:
        image = None
    finally:
        self.connection.request_path = save_request_path
    return image