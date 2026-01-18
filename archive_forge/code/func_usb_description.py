from __future__ import absolute_import
import re
import glob
import os
import os.path
def usb_description(self):
    """return a short string to name the port based on USB info"""
    if self.interface is not None:
        return '{} - {}'.format(self.product, self.interface)
    elif self.product is not None:
        return self.product
    else:
        return self.name