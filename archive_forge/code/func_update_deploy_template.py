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
def update_deploy_template(self, deploy_template, **attrs):
    """Update a deploy_template.

        :param deploy_template: Either the ID of a deploy_template,
            or an instance of
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`.
        :param dict attrs: The attributes to update on
            the deploy_template represented
            by the ``deploy_template`` parameter.

        :returns: The updated deploy_template.
        :rtype:
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`
        """
    return self._update(_deploytemplates.DeployTemplate, deploy_template, **attrs)