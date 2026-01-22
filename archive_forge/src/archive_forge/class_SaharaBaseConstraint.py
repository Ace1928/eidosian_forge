from oslo_config import cfg
from saharaclient.api import base as sahara_base
from saharaclient import client as sahara_client
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
class SaharaBaseConstraint(constraints.BaseCustomConstraint):
    expected_exceptions = (exception.EntityNotFound, exception.PhysicalResourceNameAmbiguity)
    resource_name = None

    def validate_with_client(self, client, resource_id):
        sahara_plugin = client.client_plugin(CLIENT_NAME)
        sahara_plugin.find_resource_by_name_or_id(self.resource_name, resource_id)