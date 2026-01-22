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
class GCEInstanceGroupManager(UuidMixin):
    """
    GCE Instance Groups Manager class.

    Handles 'managed' Instance Groups.
    For more information on Instance Groups, see:
    https://cloud.google.com/compute/docs/instance-groups
    """

    def __init__(self, id, name, zone, size, template, instance_group, driver, extra=None):
        """
        :param  id: Internal identifier of Instance Group.  Display only.
        :type   id: ``str``

        :param  name: The name of this Instance Group.
        :type   name: ``str``

        :param  zone: Zone in witch the Instance Group belongs
        :type   zone: :class: ``GCEZone``

        :param  size: Number of instances in this Instance Group.
        :type   size: ``int``

        :param  template: An initialized :class:``GCEInstanceTemplate``
        :type   driver: :class:``GCEInstanceTemplate``

        :param  instance_group: An initialized :class:``GCEInstanceGroup``
        :type   driver: :class:``GCEInstanceGroup``

        :param  driver: An initialized :class:``GCENodeDriver``
        :type   driver: :class:``GCENodeDriver``

        :param  extra: A dictionary of extra information.
        :type   extra: ``dict``
        """
        self.id = str(id)
        self.name = name
        self.zone = zone
        self.size = size or 0
        self.template = template
        self.instance_group = instance_group
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def destroy(self):
        """
        Destroy this Instance Group.  Destroys all instances managed by the
        Instance Group.

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_destroy_instancegroupmanager(manager=self)

    def list_managed_instances(self):
        """
        Lists all of the instances in this managed instance group.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute
        * https://www.googleapis.com/auth/compute.readonly

        :return:  ``list`` of ``dict`` containing instance URI and
                  currentAction. See
                  ex_instancegroupmanager_list_managed_instances for
                  more details.
        :rtype: ``list``
        """
        return self.driver.ex_instancegroupmanager_list_managed_instances(manager=self)

    def set_instancetemplate(self, instancetemplate):
        """
        Set the Instance Template for this Instance Group.

        :param  instancetemplate: Instance Template to set.
        :type   instancetemplate: :class:`GCEInstanceTemplate`

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_instancegroupmanager_set_instancetemplate(manager=self, instancetemplate=instancetemplate)

    def recreate_instances(self):
        """
        Recreate instances in a Managed Instance Group.

        :return:  ``list`` of ``dict`` containing instance URI and
                  currentAction. See
                  ex_instancegroupmanager_list_managed_instances for
                  more details.
        :rtype: ``list``
        """
        return self.driver.ex_instancegroupmanager_recreate_instances(manager=self)

    def delete_instances(self, node_list):
        """
        Removes one or more instances from the specified instance group,
        and delete those instances.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  node_list: List of nodes to delete.
        :type   node_list: ``list`` of :class:`Node` or ``list`` of
                           :class:`GCENode`

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_instancegroupmanager_delete_instances(manager=self, node_list=node_list)

    def resize(self, size):
        """
        Set the number of instances for this Instance Group.  An increase in
        num_instances will result in VMs being created.  A decrease will result
        in VMs being destroyed.

        :param  size: Number to instances to resize to.
        :type   size: ``int``

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_instancegroupmanager_resize(manager=self, size=size)

    def set_named_ports(self, named_ports):
        """
        Sets the named ports for the instance group controlled by this manager.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  named_ports:  Assigns a name to a port number. For example:
                              {name: "http", port: 80}  This allows the
                              system to reference ports by the assigned name
                              instead of a port number. Named ports can also
                              contain multiple ports. For example: [{name:
                              "http", port: 80},{name: "http", port: 8080}]
                              Named ports apply to all instances in this
                              instance group.
        :type   named_ports: ``list`` of {'name': ``str``, 'port`: ``int``}

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_instancegroup_set_named_ports(instancegroup=self.instance_group, named_ports=named_ports)

    def set_autohealingpolicies(self, healthcheck, initialdelaysec):
        """
        Sets the autohealing policies for the instance for the instance group
        controlled by this manager.

        :param  healthcheck: Healthcheck to add
        :type   healthcheck: :class:`GCEHealthCheck`

        :param  initialdelaysec:  The time to allow an instance to boot and
                                  applications to fully start before the first
                                  health check
        :type   initialdelaysec:  ``int``

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_instancegroupmanager_set_autohealingpolicies(manager=self, healthcheck=healthcheck, initialdelaysec=initialdelaysec)

    def __repr__(self):
        return '<GCEInstanceGroupManager name="%s" zone="%s" size="%d">' % (self.name, self.zone.name, self.size)