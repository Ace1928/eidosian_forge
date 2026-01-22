import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
class Hypothesis(object):
    """
    Represents a hypothesis about the meaning of the device identifier.
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    @abc.abstractmethod
    def match(cls, value):
        """
        Match the given string according to the hypothesis.

        The purpose of this method is to obtain a value corresponding to
        ``value`` if that is possible. It may use a regular expression, but
        in general it should just return ``value`` and let the lookup method
        sort out the rest.

        :param str value: the string to inspect
        :returns: the matched thing or None if unmatched
        :rtype: the type of lookup's key parameter or NoneType
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def lookup(cls, context, key):
        """
        Lookup the given string according to the hypothesis.

        :param Context context: the pyudev context
        :param key: a key with which to lookup the device
        :type key: the type of match's return value if not None
        :returns: a list of Devices obtained
        :rtype: frozenset of :class:`Device`
        """
        raise NotImplementedError()

    @classmethod
    def setup(cls, context):
        """
        A potentially expensive method that may allow an :class:`Hypothesis`
        to find devices more rapidly or to find a device that it would
        otherwise miss.

        :param Context context: the pyudev context
        """
        pass

    @classmethod
    def get_devices(cls, context, value):
        """
        Get any devices that may correspond to the given string.

        :param Context context: the pyudev context
        :param str value: the value to look for
        :returns: a list of devices obtained
        :rtype: set of :class:`Device`
        """
        key = cls.match(value)
        return cls.lookup(context, key) if key is not None else frozenset()