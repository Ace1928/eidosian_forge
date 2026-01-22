from __future__ import absolute_import
import re
import glob
import os
import os.path
def apply_usb_info(self):
    """update description and hwid from USB data"""
    self.description = self.usb_description()
    self.hwid = self.usb_info()