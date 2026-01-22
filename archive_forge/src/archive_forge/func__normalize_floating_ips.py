import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def _normalize_floating_ips(self, ips):
    """Normalize the structure of floating IPs

        Unfortunately, not all the Neutron floating_ip attributes are available
        with Nova and not all Nova floating_ip attributes are available with
        Neutron.
        This function extract attributes that are common to Nova and Neutron
        floating IP resource.
        If the whole structure is needed inside shade, shade provides private
        methods that returns "original" objects (e.g.
        _neutron_allocate_floating_ip)

        :param list ips: A list of Neutron floating IPs.

        :returns:
            A list of normalized dicts with the following attributes::

                [
                    {
                        "id": "this-is-a-floating-ip-id",
                        "fixed_ip_address": "192.0.2.10",
                        "floating_ip_address": "198.51.100.10",
                        "network": "this-is-a-net-or-pool-id",
                        "attached": True,
                        "status": "ACTIVE"
                    }, ...
                ]

        """
    return [self._normalize_floating_ip(ip) for ip in ips]