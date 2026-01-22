from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
def GetPwd(self):
    """Gets the coshell pwd, sets local pwd, returns the pwd, None on error."""
    pwd = self.Communicate(['printf "$PWD\\n\\n"'], quote=False)
    if len(pwd) == 1:
        try:
            os.chdir(pwd[0])
            return pwd[0]
        except OSError:
            pass
    return None