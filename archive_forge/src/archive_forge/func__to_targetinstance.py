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
def _to_targetinstance(self, targetinstance):
    """
        Return a Target Instance object from the JSON-response dictionary.

        :param  targetinstance: The dictionary describing the target instance.
        :type   targetinstance: ``dict``

        :return: Target Instance object
        :rtype:  :class:`GCETargetInstance`
        """
    node = None
    extra = {}
    extra['selfLink'] = targetinstance.get('selfLink')
    extra['description'] = targetinstance.get('description')
    extra['natPolicy'] = targetinstance.get('natPolicy')
    zone = self.ex_get_zone(targetinstance['zone'])
    if 'instance' in targetinstance:
        node_name = targetinstance['instance'].split('/')[-1]
        try:
            node = self.ex_get_node(node_name, zone)
        except ResourceNotFoundError:
            node = targetinstance['instance']
    return GCETargetInstance(id=targetinstance['id'], name=targetinstance['name'], zone=zone, node=node, driver=self, extra=extra)