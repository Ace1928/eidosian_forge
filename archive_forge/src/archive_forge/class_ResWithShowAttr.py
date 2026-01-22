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
class ResWithShowAttr(GenericResource):

    def _show_resource(self):
        return {'foo': self.name, 'Foo': self.name, 'Another': self.name}