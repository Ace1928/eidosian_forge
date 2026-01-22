from neutron_lib._i18n import _
from neutron_lib import exceptions
class NotificationError(object):

    def __init__(self, callback_id, error, cancellable=False):
        self.callback_id = callback_id
        self.error = error
        self._cancellable = cancellable

    def __str__(self):
        return 'Callback %s failed with "%s"' % (self.callback_id, self.error)

    @property
    def is_cancellable(self):
        return self._cancellable