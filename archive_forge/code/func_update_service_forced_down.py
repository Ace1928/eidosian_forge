import warnings
from openstack.block_storage.v3 import volume as _volume
from openstack.compute.v2 import aggregate as _aggregate
from openstack.compute.v2 import availability_zone
from openstack.compute.v2 import extension
from openstack.compute.v2 import flavor as _flavor
from openstack.compute.v2 import hypervisor as _hypervisor
from openstack.compute.v2 import image as _image
from openstack.compute.v2 import keypair as _keypair
from openstack.compute.v2 import limits
from openstack.compute.v2 import migration as _migration
from openstack.compute.v2 import quota_set as _quota_set
from openstack.compute.v2 import server as _server
from openstack.compute.v2 import server_action as _server_action
from openstack.compute.v2 import server_diagnostics as _server_diagnostics
from openstack.compute.v2 import server_group as _server_group
from openstack.compute.v2 import server_interface as _server_interface
from openstack.compute.v2 import server_ip
from openstack.compute.v2 import server_migration as _server_migration
from openstack.compute.v2 import server_remote_console as _src
from openstack.compute.v2 import service as _service
from openstack.compute.v2 import usage as _usage
from openstack.compute.v2 import volume_attachment as _volume_attachment
from openstack import exceptions
from openstack.identity.v3 import project as _project
from openstack.network.v2 import security_group as _sg
from openstack import proxy
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
def update_service_forced_down(self, service, host=None, binary=None, forced=True):
    """Update service forced_down information

        :param service: Either the ID of a service or a
            :class:`~openstack.compute.v2.service.Service` instance.
        :param str host: The host where service runs.
        :param str binary: The name of service.
        :param bool forced: Whether or not this service was forced down
            manually by an administrator after the service was fenced.

        :returns: Updated service instance
        :rtype: class: `~openstack.compute.v2.service.Service`
        """
    if utils.supports_microversion(self, '2.53'):
        return self.update_service(service, forced_down=forced)
    service = self._get_resource(_service.Service, service)
    if (not host or not binary) and (not service.host or not service.binary):
        raise ValueError('Either service instance should have host and binary or they should be passed')
    service.set_forced_down(self, host, binary, forced)