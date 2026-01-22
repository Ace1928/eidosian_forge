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
class GCETargetPool(UuidMixin):

    def __init__(self, id, name, region, healthchecks, nodes, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.region = region
        self.healthchecks = healthchecks
        self.nodes = nodes
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def add_node(self, node):
        """
        Add a node to this target pool.

        :param  node: Node to add
        :type   node: ``str`` or :class:`Node`

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_targetpool_add_node(targetpool=self, node=node)

    def remove_node(self, node):
        """
        Remove a node from this target pool.

        :param  node: Node to remove
        :type   node: ``str`` or :class:`Node`

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_targetpool_remove_node(targetpool=self, node=node)

    def add_healthcheck(self, healthcheck):
        """
        Add a healthcheck to this target pool.

        :param  healthcheck: Healthcheck to add
        :type   healthcheck: ``str`` or :class:`GCEHealthCheck`

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_targetpool_add_healthcheck(targetpool=self, healthcheck=healthcheck)

    def remove_healthcheck(self, healthcheck):
        """
        Remove a healthcheck from this target pool.

        :param  healthcheck: Healthcheck to remove
        :type   healthcheck: ``str`` or :class:`GCEHealthCheck`

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_targetpool_remove_healthcheck(targetpool=self, healthcheck=healthcheck)

    def set_backup_targetpool(self, backup_targetpool, failover_ratio=0.1):
        """
        Set a backup targetpool.

        :param  backup_targetpool: The existing targetpool to use for
                                   failover traffic.
        :type   backup_targetpool: :class:`GCETargetPool`

        :param  failover_ratio: The percentage of healthy VMs must fall at or
                                below this value before traffic will be sent
                                to the backup targetpool (default 0.10)
        :type   failover_ratio: ``float``

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_targetpool_set_backup_targetpool(targetpool=self, backup_targetpool=backup_targetpool, failover_ratio=failover_ratio)

    def get_health(self, node=None):
        """
        Return a hash of target pool instances and their health.

        :param  node: Optional node to specify if only a specific node's
                      health status should be returned
        :type   node: ``str``, ``Node``, or ``None``

        :return: List of hashes of nodes and their respective health
        :rtype:  ``list`` of ``dict``
        """
        return self.driver.ex_targetpool_get_health(targetpool=self, node=node)

    def destroy(self):
        """
        Destroy this Target Pool

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_destroy_targetpool(targetpool=self)

    def __repr__(self):
        return '<GCETargetPool id="{}" name="{}" region="{}">'.format(self.id, self.name, self.region.name)