from __future__ import absolute_import
import re
import glob
import os
import os.path
def usb_info(self):
    """return a string with USB related information about device"""
    return 'USB VID:PID={:04X}:{:04X}{}{}'.format(self.vid or 0, self.pid or 0, ' SER={}'.format(self.serial_number) if self.serial_number is not None else '', ' LOCATION={}'.format(self.location) if self.location is not None else '')