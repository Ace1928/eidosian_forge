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
def GetURL(self):
    if not self.known:
        return None
    if 'darwin' in sys.platform:
        ref = ['https://developer.apple.com/legacy/library/documentation', 'Darwin/Reference/ManPages/man1']
    else:
        ref = ['http://man7.org/linux/man-pages/man1']
    ref.append(self.args[0] + '.1.html')
    return '/'.join(ref)