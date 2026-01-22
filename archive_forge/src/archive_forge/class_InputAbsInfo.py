import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
class InputAbsInfo(object):
    """
    A class representing the struct input_absinfo for a given EV_ABS code.

    Any of the attributes may be set to None, those that are None are simply
    ignored by libevdev.

    :property minimum: the minimum value for this axis
    :property maximum: the maximum value for this axis
    :property fuzz: the fuzz value for this axis
    :property flat: the flat value for this axis
    :property resolution: the resolution for this axis
    :property value: the current value of this axis
    """

    def __init__(self, minimum=None, maximum=None, fuzz=None, flat=None, resolution=None, value=None):
        self.minimum = minimum
        self.maximum = maximum
        self.fuzz = fuzz
        self.flat = flat
        self.resolution = resolution
        self.value = value

    def __repr__(self):
        return 'min:{} max:{} fuzz:{} flat:{} resolution:{} value:{}'.format(self.minimum, self.maximum, self.fuzz, self.flat, self.resolution, self.value)

    def __eq__(self, other):
        return self.minimum == other.minimum and self.maximum == other.maximum and (self.value == other.value) and (self.resolution == other.resolution) and (self.fuzz == other.fuzz) and (self.flat == other.flat)