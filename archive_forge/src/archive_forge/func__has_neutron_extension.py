from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def _has_neutron_extension(self, extension_alias):
    return extension_alias in self._neutron_extensions()