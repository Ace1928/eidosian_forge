import collections
import contextlib
import datetime as dt
import itertools
import pydoc
import re
import tenacity
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import short_id
from heat.common import timeutils
from heat.engine import attributes
from heat.engine.cfn import template as cfn_tmpl
from heat.engine import clients
from heat.engine.clients import default_client_plugin
from heat.engine import environment
from heat.engine import event
from heat.engine import function
from heat.engine.hot import template as hot_tmpl
from heat.engine import node_data
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import status
from heat.engine import support
from heat.engine import sync_point
from heat.engine import template
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_objects
from heat.objects import resource_properties_data as rpd_objects
from heat.rpc import client as rpc_client
def action_handler_task(self, action, args=None, action_prefix=None):
    """A task to call the Resource subclass's handler methods for action.

        Calls the handle_<ACTION>() method for the given action and then calls
        the check_<ACTION>_complete() method with the result in a loop until it
        returns True. If the methods are not provided, the call is omitted.

        Any args provided are passed to the handler.

        If a prefix is supplied, the handler method handle_<PREFIX>_<ACTION>()
        is called instead.
        """
    args = args or []
    handler_action = action.lower()
    check = getattr(self, 'check_%s_complete' % handler_action, None)
    if action_prefix:
        handler_action = '%s_%s' % (action_prefix.lower(), handler_action)
    handler = getattr(self, 'handle_%s' % handler_action, None)
    if callable(handler):
        try:
            handler_data = handler(*args)
        except StopIteration:
            raise RuntimeError('Plugin method raised StopIteration')
        yield
        if callable(check):
            try:
                while True:
                    try:
                        done = check(handler_data)
                    except PollDelay as delay:
                        yield delay.period
                    else:
                        if done:
                            break
                        else:
                            yield
            except StopIteration:
                raise RuntimeError('Plugin method raised StopIteration')
            except Exception:
                raise
            except:
                with excutils.save_and_reraise_exception():
                    canceller = getattr(self, 'handle_%s_cancel' % handler_action, None)
                    if callable(canceller):
                        try:
                            canceller(handler_data)
                        except Exception:
                            LOG.exception('Error cancelling resource %s', action)