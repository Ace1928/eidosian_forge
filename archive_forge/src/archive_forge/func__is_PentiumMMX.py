import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_PentiumMMX(self):
    return self.is_Intel() and self.info[0]['Family'] == 5 and (self.info[0]['Model'] == 4)