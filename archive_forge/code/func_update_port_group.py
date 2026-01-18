from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def update_port_group(self, port_group, **attrs):
    """Update a port group.

        :param port_group: Either the name or the ID of a port group or
            an instance of
            :class:`~openstack.baremetal.v1.port_group.PortGroup`.
        :param dict attrs: The attributes to update on the port group
            represented by the ``port_group`` parameter.

        :returns: The updated port group.
        :rtype: :class:`~openstack.baremetal.v1.port_group.PortGroup`
        """
    return self._update(_portgroup.PortGroup, port_group, **attrs)