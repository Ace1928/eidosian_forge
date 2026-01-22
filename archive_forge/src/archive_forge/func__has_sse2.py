import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _has_sse2(self):
    if self.is_Intel():
        return self.is_Pentium4() or self.is_PentiumM() or self.is_Core2()
    elif self.is_AMD():
        return self.is_AMD64()
    else:
        return False