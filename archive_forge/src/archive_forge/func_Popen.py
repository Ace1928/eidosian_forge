from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
import subprocess
import sys
import webbrowser
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def Popen(self, args, **kwargs):
    with files.FileWriter(os.devnull) as devnull:
        kwargs.update({'stderr': devnull, 'stdout': devnull})
        return subprocess.Popen(args, **kwargs)