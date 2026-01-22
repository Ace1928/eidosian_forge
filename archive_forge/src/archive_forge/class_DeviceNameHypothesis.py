import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
class DeviceNameHypothesis(Hypothesis):
    """
    Discover the device assuming the input is a device name.

    Try every available subsystem.
    """

    @classmethod
    def find_subsystems(cls, context):
        """
        Find all subsystems in sysfs.

        :param Context context: the context
        :rtype: frozenset
        :returns: subsystems in sysfs
        """
        sys_path = context.sys_path
        dirnames = ('bus', 'class', 'subsystem')
        absnames = (os.path.join(sys_path, name) for name in dirnames)
        realnames = (d for d in absnames if os.path.isdir(d))
        return frozenset((n for d in realnames for n in os.listdir(d)))

    @classmethod
    def match(cls, value):
        """
        Match ``value`` under the assumption that it is a device name.

        :returns: the device path or None
        :rtype: str or NoneType
        """
        return value

    @classmethod
    def lookup(cls, context, key):
        """
        Lookup by the path.

        :param Context context: the context
        :param str key: the device path
        :returns: a list of matching devices
        :rtype: frozenset of :class:`Device`
        """
        func = wrap_exception(Devices.from_name)
        res = (func(context, s, key) for s in cls.find_subsystems(context))
        return frozenset((r for r in res if r is not None))