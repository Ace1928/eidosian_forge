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
class ResourceWithProps(GenericResource):
    properties_schema = {'Foo': properties.Schema(properties.Schema.STRING), 'FooInt': properties.Schema(properties.Schema.INTEGER)}
    atomic_key = None