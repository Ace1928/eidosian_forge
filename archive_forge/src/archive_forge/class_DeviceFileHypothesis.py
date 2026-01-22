import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
class DeviceFileHypothesis(Hypothesis):
    """
    Discover the device assuming the value is some portion of a device file.

    The device file may be a link to a device node.
    """
    _LINK_DIRS = ['/dev', '/dev/disk/by-id', '/dev/disk/by-label', '/dev/disk/by-partlabel', '/dev/disk/by-partuuid', '/dev/disk/by-path', '/dev/disk/by-uuid', '/dev/input/by-path', '/dev/mapper', '/dev/md', '/dev/vg']

    @classmethod
    def get_link_dirs(cls, context):
        """
        Get all directories that may contain links to device nodes.

        This method checks the device links of every device, so it is very
        expensive.

        :param Context context: the context
        :returns: a sorted list of directories that contain device links
        :rtype: list
        """
        devices = context.list_devices()
        devices_with_links = (d for d in devices if list(d.device_links))
        links = (l for d in devices_with_links for l in d.device_links)
        return sorted(set((os.path.dirname(l) for l in links)))

    @classmethod
    def setup(cls, context):
        """
        Set the link directories to be used when discovering by file.

        Uses `get_link_dirs`, so is as expensive as it is.

        :param Context context: the context
        """
        cls._LINK_DIRS = cls.get_link_dirs(context)

    @classmethod
    def match(cls, value):
        return value

    @classmethod
    def lookup(cls, context, key):
        """
        Lookup the device under the assumption that the key is part of
        the name of a device file.

        :param Context context: the context
        :param str key: a portion of the device file name

        It is assumed that either it is the whole name of the device file
        or it is the basename.

        A device file may be a device node or a device link.
        """
        func = wrap_exception(Devices.from_device_file)
        if '/' in key:
            device = func(context, key)
            return frozenset((device,)) if device is not None else frozenset()
        files = (os.path.join(ld, key) for ld in cls._LINK_DIRS)
        devices = (func(context, f) for f in files)
        return frozenset((d for d in devices if d is not None))