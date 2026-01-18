from openstack import exceptions
from openstack.orchestration.util import template_utils
from openstack.orchestration.v1 import resource as _resource
from openstack.orchestration.v1 import software_config as _sc
from openstack.orchestration.v1 import software_deployment as _sd
from openstack.orchestration.v1 import stack as _stack
from openstack.orchestration.v1 import stack_environment as _stack_environment
from openstack.orchestration.v1 import stack_event as _stack_event
from openstack.orchestration.v1 import stack_files as _stack_files
from openstack.orchestration.v1 import stack_template as _stack_template
from openstack.orchestration.v1 import template as _template
from openstack import proxy
from openstack import resource
def update_software_deployment(self, software_deployment, **attrs):
    """Update a software deployment

        :param server: Either the ID of a software deployment or an instance of
            :class:`~openstack.orchestration.v1.software_deployment.SoftwareDeployment`
        :param dict attrs: The attributes to update on the software deployment
            represented by ``software_deployment``.

        :returns: The updated software deployment
        :rtype:
            :class:`~openstack.orchestration.v1.software_deployment.SoftwareDeployment`
        """
    return self._update(_sd.SoftwareDeployment, software_deployment, **attrs)