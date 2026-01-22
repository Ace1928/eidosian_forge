import eventlet.queue
import functools
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import resource as resource_objects
from heat.rpc import api as rpc_api
from heat.rpc import listener_client
class CancelOperation(BaseException):
    """Exception to cancel an in-progress operation on a resource.

    This exception is raised when operations on a resource are cancelled.
    """

    def __init__(self):
        return super(CancelOperation, self).__init__('user triggered cancel')