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
class ResourceWithRestoreType(ResWithComplexPropsAndAttrs):

    def handle_restore(self, defn, data):
        props = dict(((key, value) for key, value in self.properties.data.items() if value is not None))
        value = data['resource_data']['a_string']
        props['a_string'] = value
        return defn.freeze(properties=props)

    def handle_delete_snapshot(self, snapshot):
        return snapshot['resource_data'].get('a_string')