import collections
from oslo_log import log as logging
from oslo_utils import reflection
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import priority_group
from neutron_lib.db import utils as db_utils
def _notify_loop(self, resource, event, trigger, payload):
    """The notification loop."""
    errors = []
    callbacks = []
    for pri_callbacks in self._callbacks[resource].get(event, []):
        for cb_id, cb_method in pri_callbacks.pri_callbacks.items():
            cb = Callback(cb_id, cb_method, pri_callbacks.cancellable)
            callbacks.append(cb)
    resource_id = getattr(payload, 'resource_id', None)
    LOG.debug('Publish callbacks %s for %s (%s), %s', [c.id for c in callbacks], resource, resource_id, event)
    for callback in callbacks:
        try:
            callback.method(resource, event, trigger, payload=payload)
        except Exception as e:
            if not (events.is_cancellable_event(event) or callback.cancellable):
                LOG.exception('Error during notification for %(callback)s %(resource)s, %(event)s', {'callback': callback.id, 'resource': resource, 'event': event})
            else:
                LOG.debug('Callback %(callback)s raised %(error)s', {'callback': callback.id, 'error': e})
            errors.append(exceptions.NotificationError(callback.id, e, cancellable=callback.cancellable))
    return errors