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
class GCEInstanceTemplate(UuidMixin):
    """Represents a machine configuration used in creating Instance Groups."""

    def __init__(self, id, name, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def __repr__(self):
        return '<GCEInstanceTemplate id="{}" name="{}" machineType="{}">'.format(self.id, self.name, self.extra['properties'].get('machineType', 'UNKNOWN'))

    def destroy(self):
        """
        Destroy this InstanceTemplate.

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_destroy_instancetemplate(instancetemplate=self)