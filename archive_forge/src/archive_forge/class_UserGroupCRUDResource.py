import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
class UserGroupCRUDResource(flask_restful.Resource):

    @staticmethod
    def _build_enforcement_target_attr(user_id, group_id):
        target = {}
        try:
            target['group'] = PROVIDERS.identity_api.get_group(group_id)
        except exception.GroupNotFound:
            pass
        try:
            target['user'] = PROVIDERS.identity_api.get_user(user_id)
        except exception.UserNotFound:
            pass
        return target

    def get(self, group_id, user_id):
        """Check if a user is in a group.

        GET/HEAD /groups/{group_id}/users/{user_id}
        """
        ENFORCER.enforce_call(action='identity:check_user_in_group', build_target=functools.partial(self._build_enforcement_target_attr, user_id, group_id))
        PROVIDERS.identity_api.check_user_in_group(user_id, group_id)
        return (None, http.client.NO_CONTENT)

    def put(self, group_id, user_id):
        """Add user to group.

        PUT /groups/{group_id}/users/{user_id}
        """
        ENFORCER.enforce_call(action='identity:add_user_to_group', build_target=functools.partial(self._build_enforcement_target_attr, user_id, group_id))
        PROVIDERS.identity_api.add_user_to_group(user_id, group_id, initiator=notifications.build_audit_initiator())
        return (None, http.client.NO_CONTENT)

    def delete(self, group_id, user_id):
        """Remove user from group.

        DELETE /groups/{group_id}/users/{user_id}
        """
        ENFORCER.enforce_call(action='identity:remove_user_from_group', build_target=functools.partial(self._build_enforcement_target_attr, user_id, group_id))
        PROVIDERS.identity_api.remove_user_from_group(user_id, group_id, initiator=notifications.build_audit_initiator())
        return (None, http.client.NO_CONTENT)