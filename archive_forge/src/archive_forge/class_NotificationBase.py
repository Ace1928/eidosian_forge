import abc
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob
from glance.common import exception
from glance.common import timeutils
from glance.domain import proxy as domain_proxy
from glance.i18n import _, _LE
class NotificationBase(object):

    def get_payload(self, obj):
        return {}

    def send_notification(self, notification_id, obj, extra_payload=None, backend=None):
        payload = self.get_payload(obj)
        if extra_payload is not None:
            payload.update(extra_payload)
        if backend:
            payload['backend'] = backend
        _send_notification(self.notifier.info, notification_id, payload)