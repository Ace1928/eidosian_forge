from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def add_router_interface(self, router, subnet_id=None, port_id=None):
    """Attach a subnet to an internal router interface.

        Either a subnet ID or port ID must be specified for the internal
        interface. Supplying both will result in an error.

        :param dict router: The dict object of the router being changed
        :param string subnet_id: The ID of the subnet to use for the interface
        :param string port_id: The ID of the port to use for the interface

        :returns: The raw response body from the request.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    return self.network.add_interface_to_router(router=router, subnet_id=subnet_id, port_id=port_id)