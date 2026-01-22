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
class GCEUrlMap(UuidMixin):
    """A GCE URL Map."""

    def __init__(self, id, name, default_service, host_rules, path_matchers, tests, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.default_service = default_service
        self.host_rules = host_rules or []
        self.path_matchers = path_matchers or []
        self.tests = tests or []
        self.driver = driver
        self.extra = extra or {}
        UuidMixin.__init__(self)

    def __repr__(self):
        return '<GCEUrlMap id="{}" name="{}">'.format(self.id, self.name)

    def destroy(self):
        """
        Destroy this URL Map

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_destroy_urlmap(urlmap=self)