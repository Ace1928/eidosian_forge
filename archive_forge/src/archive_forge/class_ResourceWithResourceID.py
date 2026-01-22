import collections
from oslo_log import log as logging
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import signal_responder
from heat.engine.resources import stack_resource
from heat.engine.resources import stack_user
from heat.engine import support
class ResourceWithResourceID(GenericResource):
    properties_schema = {'ID': properties.Schema(properties.Schema.STRING)}

    def handle_create(self):
        super(ResourceWithResourceID, self).handle_create()
        self.resource_id_set(self.properties.get('ID'))

    def handle_delete(self):
        self.mock_resource_id(self.resource_id)

    def mock_resource_id(self, resource_id):
        pass