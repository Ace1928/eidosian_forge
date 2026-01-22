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
class EventController(object):
    """WSGI controller for Events in Heat v1 API.

    Implements the API actions.
    """
    REQUEST_SCOPE = 'events'

    def __init__(self, options):
        self.options = options
        self.rpc_client = rpc_client.EngineClient()

    def _event_list(self, req, identity, detail=False, filters=None, limit=None, marker=None, sort_keys=None, sort_dir=None, nested_depth=None):
        events = self.rpc_client.list_events(req.context, identity, filters=filters, limit=limit, marker=marker, sort_keys=sort_keys, sort_dir=sort_dir, nested_depth=nested_depth)
        keys = None if detail else summary_keys
        return [format_event(req, e, keys) for e in events]

    @util.registered_identified_stack
    def index(self, req, identity, resource_name=None):
        """Lists summary information for all events."""
        param_types = {'limit': util.PARAM_TYPE_SINGLE, 'marker': util.PARAM_TYPE_SINGLE, 'sort_dir': util.PARAM_TYPE_SINGLE, 'sort_keys': util.PARAM_TYPE_MULTI, 'nested_depth': util.PARAM_TYPE_SINGLE}
        filter_param_types = {'resource_status': util.PARAM_TYPE_MIXED, 'resource_action': util.PARAM_TYPE_MIXED, 'resource_name': util.PARAM_TYPE_MIXED, 'resource_type': util.PARAM_TYPE_MIXED}
        params = util.get_allowed_params(req.params, param_types)
        filter_params = util.get_allowed_params(req.params, filter_param_types)
        int_params = (rpc_api.PARAM_LIMIT, rpc_api.PARAM_NESTED_DEPTH)
        try:
            for key in int_params:
                if key in params:
                    params[key] = param_utils.extract_int(key, params[key], allow_zero=True)
        except ValueError as e:
            raise exc.HTTPBadRequest(str(e))
        if resource_name is None:
            if not filter_params:
                filter_params = None
        else:
            filter_params['resource_name'] = resource_name
        events = self._event_list(req, identity, filters=filter_params, **params)
        if not events and resource_name is not None:
            msg = _('No events found for resource %s') % resource_name
            raise exc.HTTPNotFound(msg)
        return {'events': events}

    @util.registered_identified_stack
    def show(self, req, identity, resource_name, event_id):
        """Gets detailed information for an event."""
        filters = {'resource_name': resource_name, 'uuid': event_id}
        events = self._event_list(req, identity, filters=filters, detail=True)
        if not events:
            raise exc.HTTPNotFound(_('No event %s found') % event_id)
        return {'event': events[0]}