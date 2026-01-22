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
def _to_targethttpsproxy(self, targethttpsproxy):
    """
        Return the TargetHttpsProxy object from the JSON-response.

        :param  targethttpsproxy:  Dictionary describing TargetHttpsProxy
        :type   targethttpsproxy: ``dict``

        :return:  Return TargetHttpsProxy object.
        :rtype: :class:`GCETargetHttpsProxy`
        """
    extra = {}
    if 'description' in targethttpsproxy:
        extra['description'] = targethttpsproxy['description']
    extra['selfLink'] = targethttpsproxy['selfLink']
    sslcertificates = [self._get_object_by_kind(x) for x in targethttpsproxy.get('sslCertificates', [])]
    obj_name = self._get_components_from_path(targethttpsproxy['urlMap'])['name']
    urlmap = self.ex_get_urlmap(obj_name)
    return GCETargetHttpsProxy(id=targethttpsproxy['id'], name=targethttpsproxy['name'], sslcertificates=sslcertificates, urlmap=urlmap, driver=self, extra=extra)