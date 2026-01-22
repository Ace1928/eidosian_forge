import flask
import flask_restful
from oslo_utils import timeutils
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class OSRevokeResource(flask_restful.Resource):

    def get(self):
        ENFORCER.enforce_call(action='identity:list_revoke_events')
        since = flask.request.args.get('since')
        last_fetch = None
        if since:
            try:
                last_fetch = timeutils.normalize_time(timeutils.parse_isotime(since))
            except ValueError:
                raise exception.ValidationError(message=_('invalidate date format %s') % since)
        events = PROVIDERS.revoke_api.list_events(last_fetch=last_fetch)
        response = {'events': [event.to_dict() for event in events], 'links': {'next': None, 'self': ks_flask.base_url(path='/OS-REVOKE/events'), 'previous': None}}
        return response