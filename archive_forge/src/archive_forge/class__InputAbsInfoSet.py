import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
class _InputAbsInfoSet:

    def __init__(self, parent_device):
        self._device = parent_device

    def __getitem__(self, code):
        assert code.type == libevdev.EV_ABS
        r = self._device._libevdev.absinfo(code.value)
        if r is None:
            return r
        return InputAbsInfo(r['minimum'], r['maximum'], r['fuzz'], r['flat'], r['resolution'], r['value'])

    def __setitem__(self, code, absinfo):
        assert code.type == libevdev.EV_ABS
        if not self._device.has(code):
            raise InvalidArgumentException('Device does not have event code')
        data = {}
        if absinfo.minimum is not None:
            data['minimum'] = absinfo.minimum
        if absinfo.maximum is not None:
            data['maximum'] = absinfo.maximum
        if absinfo.fuzz is not None:
            data['fuzz'] = absinfo.fuzz
        if absinfo.flat is not None:
            data['flat'] = absinfo.flat
        if absinfo.resolution is not None:
            data['resolution'] = absinfo.resolution
        if absinfo.value is not None:
            data['value'] = absinfo.value
        self._device._libevdev.absinfo(code.value, data)