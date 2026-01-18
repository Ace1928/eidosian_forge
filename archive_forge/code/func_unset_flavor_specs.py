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
def unset_flavor_specs(self, flavor_id, keys):
    """Delete extra specs from a flavor

        :param string flavor_id: ID of the flavor to update.
        :param keys: List of spec keys to delete.

        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        :raises: :class:`~openstack.exceptions.BadRequestException` if flavor
            ID is not found.
        """
    for key in keys:
        self.compute.delete_flavor_extra_specs_property(flavor_id, key)