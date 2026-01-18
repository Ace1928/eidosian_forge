import logging
import threading
import enum
from oslo_utils import reflection
from glance_store import exceptions
from glance_store.i18n import _LW
def unset_capabilities(self, *dynamic_capabilites):
    """
        Unset dynamic storage capabilities.

        :param dynamic_capabilites: dynamic storage capability(s).
        """
    caps = 0
    for cap in dynamic_capabilites:
        caps |= int(cap)
    self._capabilities &= ~caps