import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def DiscoverLocalHIDU2FDevices(selector=HidUsageSelector):
    for d in hid.Enumerate():
        if selector(d):
            try:
                dev = hid.Open(d['path'])
                yield UsbHidTransport(dev)
            except OSError:
                pass