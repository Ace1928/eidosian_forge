from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
import os_service_types
from heat.common import config
from heat.engine.clients import client_plugin
from heat.engine import constraints
import heat.version
class SegmentConstraint(constraints.BaseCustomConstraint):
    expected_exceptions = (exceptions.ResourceNotFound, exceptions.DuplicateResource)

    def validate_with_client(self, client, value):
        sdk_plugin = client.client_plugin(CLIENT_NAME)
        sdk_plugin.find_network_segment(value)