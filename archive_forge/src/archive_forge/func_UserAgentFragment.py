from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
def UserAgentFragment(self):
    """Generates the fragment of the User-Agent that represents the OS.

    Examples:
      (Linux 3.2.5-gg1236)
      (Windows NT 6.1.7601)
      (Macintosh; PPC Mac OS X 12.4.0)
      (Macintosh; Intel Mac OS X 12.4.0)

    Returns:
      str, The fragment of the User-Agent string.
    """
    if self.operating_system == OperatingSystem.LINUX:
        return '({name} {version})'.format(name=self.operating_system.name, version=self.operating_system.version)
    elif self.operating_system == OperatingSystem.WINDOWS:
        return '({name} NT {version})'.format(name=self.operating_system.name, version=self.operating_system.version)
    elif self.operating_system == OperatingSystem.MACOSX:
        format_string = '(Macintosh; {name} Mac OS X {version})'
        arch_string = self.architecture.name if self.architecture == Architecture.ppc else 'Intel'
        return format_string.format(name=arch_string, version=self.operating_system.version)
    else:
        return '()'