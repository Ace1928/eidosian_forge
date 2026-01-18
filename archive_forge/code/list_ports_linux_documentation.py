from __future__ import absolute_import
import glob
import os
from serial.tools import list_ports_common
        Helper function to read a single line from a file.
        One or more parameters are allowed, they are joined with os.path.join.
        Returns None on errors..
        