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
def _to_urlmap(self, urlmap):
    """
        Return a UrlMap object from the JSON-response dictionary.

        :param  zone: The dictionary describing the url-map.
        :type   zone: ``dict``

        :return: UrlMap object
        :rtype: :class:`GCEUrlMap`
        """
    extra = {k: urlmap.get(k) for k in ('creationTimestamp', 'description', 'fingerprint', 'selfLink')}
    default_service = self._get_object_by_kind(urlmap.get('defaultService'))
    host_rules = urlmap.get('hostRules', [])
    path_matchers = urlmap.get('pathMatchers', [])
    tests = urlmap.get('tests', [])
    return GCEUrlMap(id=urlmap['id'], name=urlmap['name'], default_service=default_service, host_rules=host_rules, path_matchers=path_matchers, tests=tests, driver=self, extra=extra)