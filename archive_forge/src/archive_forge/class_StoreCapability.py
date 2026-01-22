import logging
import threading
import enum
from oslo_utils import reflection
from glance_store import exceptions
from glance_store.i18n import _LW
class StoreCapability(object):

    def __init__(self):
        self._capabilities = getattr(self.__class__, '_CAPABILITIES', 0)

    @property
    def capabilities(self):
        return self._capabilities

    @staticmethod
    def contains(x, y):
        return x & y == y

    def update_capabilities(self):
        """
        Update dynamic storage capabilities based on current
        driver configuration and backend status when needed.

        As a hook, the function will be triggered in two cases:
        calling once after store driver get configured, it was
        used to update dynamic storage capabilities based on
        current driver configuration, or calling when the
        capabilities checking of an operation failed every time,
        this was used to refresh dynamic storage capabilities
        based on backend status then.

        This function shouldn't raise any exception out.
        """
        LOG.debug("Store %s doesn't support updating dynamic storage capabilities. Please overwrite 'update_capabilities' method of the store to implement updating logics if needed." % reflection.get_class_name(self))

    def is_capable(self, *capabilities):
        """
        Check if requested capability(s) are supported by
        current driver instance.

        :param capabilities: required capability(s).
        """
        caps = 0
        for cap in capabilities:
            caps |= int(cap)
        return self.contains(self.capabilities, caps)

    def set_capabilities(self, *dynamic_capabilites):
        """
        Set dynamic storage capabilities based on current
        driver configuration and backend status.

        :param dynamic_capabilites: dynamic storage capability(s).
        """
        for cap in dynamic_capabilites:
            self._capabilities |= int(cap)

    def unset_capabilities(self, *dynamic_capabilites):
        """
        Unset dynamic storage capabilities.

        :param dynamic_capabilites: dynamic storage capability(s).
        """
        caps = 0
        for cap in dynamic_capabilites:
            caps |= int(cap)
        self._capabilities &= ~caps