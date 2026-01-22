import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_utils import timeutils
from urllib import parse as urlparse
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.oauth1 import core as oauth1
from keystone.oauth1 import schema
from keystone.oauth1 import validator
from keystone.server import flask as ks_flask
class ConsumerResource(ks_flask.ResourceBase):
    collection_key = 'consumers'
    member_key = 'consumer'
    api_prefix = '/OS-OAUTH1'
    json_home_resource_rel_func = _build_resource_relation
    json_home_parameter_rel_func = _build_parameter_relation

    def _list_consumers(self):
        ENFORCER.enforce_call(action='identity:list_consumers')
        return self.wrap_collection(PROVIDERS.oauth_api.list_consumers())

    def _get_consumer(self, consumer_id):
        ENFORCER.enforce_call(action='identity:get_consumer')
        return self.wrap_member(PROVIDERS.oauth_api.get_consumer(consumer_id))

    def get(self, consumer_id=None):
        if consumer_id is None:
            return self._list_consumers()
        return self._get_consumer(consumer_id)

    def post(self):
        ENFORCER.enforce_call(action='identity:create_consumer')
        consumer = (flask.request.get_json(force=True, silent=True) or {}).get('consumer', {})
        consumer = self._normalize_dict(consumer)
        validation.lazy_validate(schema.consumer_create, consumer)
        consumer = self._assign_unique_id(consumer)
        ref = PROVIDERS.oauth_api.create_consumer(consumer, initiator=self.audit_initiator)
        return (self.wrap_member(ref), http.client.CREATED)

    def delete(self, consumer_id):
        ENFORCER.enforce_call(action='identity:delete_consumer')
        reason = 'Invalidating token cache because consumer %(consumer_id)s has been deleted. Authorization for users with OAuth tokens will be recalculated and enforced accordingly the next time they authenticate or validate a token.' % {'consumer_id': consumer_id}
        notifications.invalidate_token_cache_notification(reason)
        PROVIDERS.oauth_api.delete_consumer(consumer_id, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)

    def patch(self, consumer_id):
        ENFORCER.enforce_call(action='identity:update_consumer')
        consumer = (flask.request.get_json(force=True, silent=True) or {}).get('consumer', {})
        validation.lazy_validate(schema.consumer_update, consumer)
        consumer = self._normalize_dict(consumer)
        self._require_matching_id(consumer)
        ref = PROVIDERS.oauth_api.update_consumer(consumer_id, consumer, initiator=self.audit_initiator)
        return self.wrap_member(ref)