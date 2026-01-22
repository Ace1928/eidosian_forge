from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
def InstallMacPythonMessage(self):
    if OperatingSystem.Current() != OperatingSystem.MACOSX:
        return ''
    return '\nTo reinstall gcloud, run:\n    $ gcloud components reinstall\n\nThis will also prompt to install a compatible version of Python.'