import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class _KeystoneExceptionMeta(type):
    """Automatically Register the Exceptions in 'KEYSTONE_API_EXCEPTIONS' list.

    The `KEYSTONE_API_EXCEPTIONS` list is utilized by flask to register a
    handler to emit sane details when the exception occurs.
    """

    def __new__(mcs, name, bases, class_dict):
        """Create a new instance and register with KEYSTONE_API_EXCEPTIONS."""
        cls = type.__new__(mcs, name, bases, class_dict)
        KEYSTONE_API_EXCEPTIONS.add(cls)
        return cls