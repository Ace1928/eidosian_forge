import itertools
from webob import exc
from heat.api.openstack.v1 import util
from heat.common.i18n import _
from heat.common import identifier
from heat.common import param_utils
from heat.common import serializers
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
def _event_list(self, req, identity, detail=False, filters=None, limit=None, marker=None, sort_keys=None, sort_dir=None, nested_depth=None):
    events = self.rpc_client.list_events(req.context, identity, filters=filters, limit=limit, marker=marker, sort_keys=sort_keys, sort_dir=sort_dir, nested_depth=nested_depth)
    keys = None if detail else summary_keys
    return [format_event(req, e, keys) for e in events]