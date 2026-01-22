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
def _to_autoscaler(self, autoscaler):
    """
        Return an Autoscaler object from the JSON-response.

        :param  autoscaler: dictionary describing the Autoscaler.
        :type   autoscaler: ``dict``

        :return: Autoscaler object.
        :rtype:  :class:`GCEAutoscaler`
        """
    extra = {}
    extra['selfLink'] = autoscaler.get('selfLink')
    extra['description'] = autoscaler.get('description')
    zone = self.ex_get_zone(autoscaler.get('zone'))
    ig_name = self._get_components_from_path(autoscaler.get('target'))['name']
    target = self.ex_get_instancegroupmanager(ig_name, zone)
    return GCEAutoscaler(id=autoscaler['id'], name=autoscaler['name'], zone=zone, target=target, policy=autoscaler['autoscalingPolicy'], driver=self, extra=extra)