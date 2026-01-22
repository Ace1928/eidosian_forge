from keystone.common import cache
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import revoke_model
from keystone import notifications
def _register_listeners(self):
    callbacks = {notifications.ACTIONS.deleted: [['OS-TRUST:trust', self._trust_callback], ['OS-OAUTH1:consumer', self._consumer_callback], ['user', self._user_callback], ['project', self._project_callback]], notifications.ACTIONS.disabled: [['user', self._user_callback]], notifications.ACTIONS.internal: [[notifications.PERSIST_REVOCATION_EVENT_FOR_USER, self._user_callback]]}
    for event, cb_info in callbacks.items():
        for resource_type, callback_fns in cb_info:
            notifications.register_event_callback(event, resource_type, callback_fns)