import fnmatch
import logging
from oslo_config import cfg
from stevedore import dispatch
import yaml
from oslo_messaging.notify import notifier
def _call_notify(self, ext, context, message, priority, retry, accepted_drivers):
    """Emit the notification.
        """
    LOG.info("Routing '%(event)s' notification to '%(driver)s' driver", {'event': message.get('event_type'), 'driver': ext.name})
    ext.obj.notify(context, message, priority, retry)