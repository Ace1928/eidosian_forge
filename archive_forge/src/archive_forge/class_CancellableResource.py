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
class CancellableResource(GenericResource):

    def check_create_complete(self, cookie):
        return True

    def handle_create_cancel(self, cookie):
        LOG.warning('Cancelling create generic resource (Type "%s")', self.type())

    def check_update_complete(self, cookie):
        return True

    def handle_update_cancel(self, cookie):
        LOG.warning('Cancelling update generic resource (Type "%s")', self.type())