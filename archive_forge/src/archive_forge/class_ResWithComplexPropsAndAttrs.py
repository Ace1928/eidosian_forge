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
class ResWithComplexPropsAndAttrs(ResWithStringPropAndAttr):
    properties_schema = {'a_string': properties.Schema(properties.Schema.STRING), 'a_list': properties.Schema(properties.Schema.LIST), 'a_map': properties.Schema(properties.Schema.MAP), 'an_int': properties.Schema(properties.Schema.INTEGER)}
    attributes_schema = {'list': attributes.Schema('A list'), 'map': attributes.Schema('A map'), 'string': attributes.Schema('A string')}
    update_allowed_properties = ('an_int',)

    def _resolve_attribute(self, name):
        try:
            return self.properties['a_%s' % name]
        except KeyError:
            return None