from abc import ABCMeta
from abc import abstractmethod
import logging
import sys
import threading
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_messaging import _utils as utils
from oslo_messaging import dispatcher
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import server as msg_server
from oslo_messaging import target as msg_target
def _watchdog(self, event, incoming):
    try:
        client_timeout = int(incoming.client_timeout)
        cm_heartbeat_interval = client_timeout / 2
    except ValueError:
        client_timeout = cm_heartbeat_interval = 0
    if cm_heartbeat_interval < 1:
        LOG.warning('Client provided an invalid timeout value of %r' % incoming.client_timeout)
        return
    while not event.wait(cm_heartbeat_interval):
        LOG.debug('Sending call-monitor heartbeat for active call to %(method)s (interval=%(interval)i)' % {'method': incoming.message.get('method'), 'interval': cm_heartbeat_interval})
        try:
            incoming.heartbeat()
        except Exception as exc:
            LOG.debug('Call-monitor heartbeat failed: %(exc)s' % {'exc': exc})
            break