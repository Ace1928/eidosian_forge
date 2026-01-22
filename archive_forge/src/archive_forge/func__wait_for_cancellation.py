import eventlet.queue
import functools
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import excutils
from oslo_utils import uuidutils
from osprofiler import profiler
from heat.common import context
from heat.common import messaging as rpc_messaging
from heat.db import api as db_api
from heat.engine import check_resource
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import stack as stack_objects
from heat.rpc import api as rpc_api
from heat.rpc import worker_client as rpc_client
def _wait_for_cancellation(stack, wait=5):
    retries = CANCEL_RETRIES
    while retries > 0:
        retries -= 1
        eventlet.sleep(wait)
        engines = db_api.engine_get_all_locked_by_stack(stack.context, stack.id)
        if not engines:
            return True
    return False