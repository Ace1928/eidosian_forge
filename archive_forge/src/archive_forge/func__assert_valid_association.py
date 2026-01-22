from oslo_log import log
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _assert_valid_association(self, endpoint_id, service_id, region_id):
    """Assert that the association is supported.

        There are three types of association supported:

        - Endpoint (in which case service and region must be None)
        - Service and region (in which endpoint must be None)
        - Service (in which case endpoint and region must be None)

        """
    if endpoint_id is not None and service_id is None and (region_id is None):
        return
    if service_id is not None and region_id is not None and (endpoint_id is None):
        return
    if service_id is not None and endpoint_id is None and (region_id is None):
        return
    raise exception.InvalidPolicyAssociation(endpoint_id=endpoint_id, service_id=service_id, region_id=region_id)