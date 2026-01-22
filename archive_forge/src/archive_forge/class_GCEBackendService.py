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
class GCEBackendService(UuidMixin):
    """A GCE Backend Service."""

    def __init__(self, id, name, backends, healthchecks, port, port_name, protocol, timeout, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.backends = backends or []
        self.healthchecks = healthchecks or []
        self.port = port
        self.port_name = port_name
        self.protocol = protocol
        self.timeout = timeout
        self.driver = driver
        self.extra = extra or {}
        UuidMixin.__init__(self)

    def __repr__(self):
        return '<GCEBackendService id="{}" name="{}">'.format(self.id, self.name)

    def destroy(self):
        """
        Destroy this Backend Service.

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_destroy_backendservice(backendservice=self)