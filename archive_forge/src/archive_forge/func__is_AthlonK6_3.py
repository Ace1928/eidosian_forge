import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_AthlonK6_3(self):
    return self._is_AMD() and self.info[0]['model'] == '3'