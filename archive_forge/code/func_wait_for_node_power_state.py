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
def wait_for_node_power_state(self, node, expected_state, timeout=None):
    """Wait for the node to reach the power state.

        :param node: The value can be the name or ID of a node or a
            :class:`~openstack.baremetal.v1.node.Node` instance.
        :param timeout: How much (in seconds) to wait for the target state
            to be reached. The value of ``None`` (the default) means
            no timeout.

        :returns: The updated :class:`~openstack.baremetal.v1.node.Node`
        """
    res = self._get_resource(_node.Node, node)
    return res.wait_for_power_state(self, expected_state, timeout=timeout)