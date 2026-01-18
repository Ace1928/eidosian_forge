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
def validate_node(self, node, required=('boot', 'deploy', 'power')):
    """Validate required information on a node.

        :param node: The value can be either the name or ID of a node or
            a :class:`~openstack.baremetal.v1.node.Node` instance.
        :param required: List of interfaces that are required to pass
            validation. The default value is the list of minimum required
            interfaces for provisioning.

        :return: dict mapping interface names to
            :class:`~openstack.baremetal.v1.node.ValidationResult` objects.
        :raises: :exc:`~openstack.exceptions.ValidationException` if validation
            fails for a required interface.
        """
    res = self._get_resource(_node.Node, node)
    return res.validate(self, required=required)