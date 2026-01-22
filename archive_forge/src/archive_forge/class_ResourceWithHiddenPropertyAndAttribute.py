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
class ResourceWithHiddenPropertyAndAttribute(GenericResource):
    properties_schema = {'supported': properties.Schema(properties.Schema.LIST, 'Supported property.'), 'hidden': properties.Schema(properties.Schema.LIST, 'Hidden property.', support_status=support.SupportStatus(status=support.HIDDEN))}
    attributes_schema = {'supported': attributes.Schema(type=attributes.Schema.STRING, description='Supported attribute.'), 'hidden': attributes.Schema(type=attributes.Schema.STRING, description='Hidden attribute', support_status=support.SupportStatus(status=support.HIDDEN))}