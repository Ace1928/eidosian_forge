from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
def update_transformations(self):
    """take list of transformation classes and instantiate them for rx and tx"""
    transformations = [EOL_TRANSFORMATIONS[self.eol]] + [TRANSFORMATIONS[f] for f in self.filters]
    self.tx_transformations = [t() for t in transformations]
    self.rx_transformations = list(reversed(self.tx_transformations))