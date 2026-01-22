import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_Intel(self):
    return self.info[0]['VendorIdentifier'] == 'GenuineIntel'