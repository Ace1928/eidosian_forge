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
def _is_notification_enabled(notification):
    disabled_notifications = CONF.disabled_notifications
    notification_group = _get_notification_group(notification)
    notifications = (notification, notification_group)
    for disabled_notification in disabled_notifications:
        if disabled_notification in notifications:
            return False
    return True