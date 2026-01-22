import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def SendBlink(self, length):
    return self.InternalExchange(UsbHidTransport.U2FHID_PROMPT, bytearray([length]))