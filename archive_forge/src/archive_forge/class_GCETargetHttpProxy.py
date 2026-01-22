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
class GCETargetHttpProxy(UuidMixin):

    def __init__(self, id, name, urlmap, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.urlmap = urlmap
        self.driver = driver
        self.extra = extra or {}
        UuidMixin.__init__(self)

    def __repr__(self):
        return '<GCETargetHttpProxy id="{}" name="{}">'.format(self.id, self.name)

    def destroy(self):
        """
        Destroy this Target HTTP Proxy.

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_destroy_targethttpproxy(targethttpproxy=self)