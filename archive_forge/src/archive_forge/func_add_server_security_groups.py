import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def add_server_security_groups(self, server, security_groups):
    """Add security groups to a server.

        Add existing security groups to an existing server. If the security
        groups are already present on the server this will continue unaffected.

        :returns: False if server or security groups are undefined, True
            otherwise.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    server, security_groups = self._get_server_security_groups(server, security_groups)
    if not (server and security_groups):
        return False
    for sg in security_groups:
        self.compute.add_security_group_to_server(server, sg)
    return True