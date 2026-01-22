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
def _licenses_from_urls(self, licenses):
    """
        Convert a list of license selfLinks into a list of :class:`GCELicense`
        objects.

        :param  licenses: A list of GCE license selfLink URLs.
        :type   licenses: ``list`` of ``str``

        :return: List of :class:`GCELicense` objects.
        :rtype:  ``list``
        """
    return_list = []
    for license in licenses:
        selfLink_parts = license.split('/')
        lic_proj = selfLink_parts[6]
        lic_name = selfLink_parts[-1]
        return_list.append(self.ex_get_license(project=lic_proj, name=lic_name))
    return return_list