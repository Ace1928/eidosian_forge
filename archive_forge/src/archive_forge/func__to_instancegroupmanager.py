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
def _to_instancegroupmanager(self, manager):
    """
        Return a Instance Group Manager object from the JSON-response.

        :param  instancegroupmanager: dictionary describing the Instance
                                  Group Manager.
        :type   instancegroupmanager: ``dict``

        :return: Instance Group Manager object.
        :rtype:  :class:`GCEInstanceGroupManager`
        """
    zone = self.ex_get_zone(manager['zone'])
    extra = {}
    extra['selfLink'] = manager.get('selfLink')
    extra['description'] = manager.get('description')
    extra['currentActions'] = manager.get('currentActions')
    extra['baseInstanceName'] = manager.get('baseInstanceName')
    extra['namedPorts'] = manager.get('namedPorts', [])
    extra['autoHealingPolicies'] = manager.get('autoHealingPolicies', [])
    template_name = self._get_components_from_path(manager['instanceTemplate'])['name']
    template = self.ex_get_instancetemplate(template_name)
    ig_name = self._get_components_from_path(manager['instanceGroup'])['name']
    instance_group = self.ex_get_instancegroup(ig_name, zone)
    return GCEInstanceGroupManager(id=manager['id'], name=manager['name'], zone=zone, size=manager['targetSize'], instance_group=instance_group, template=template, driver=self, extra=extra)