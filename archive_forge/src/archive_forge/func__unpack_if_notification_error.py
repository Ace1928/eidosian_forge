from neutron_lib._i18n import _
from neutron_lib import exceptions
@staticmethod
def _unpack_if_notification_error(exc):
    if isinstance(exc, NotificationError):
        return exc.error
    return exc