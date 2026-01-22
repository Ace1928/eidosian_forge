import os
import warnings
import xml.dom.minidom
from .base import (
def _grab_xml(self, module):
    cmd = CommandLine(command='Slicer3', resource_monitor=False, args='--launch %s --xml' % module)
    ret = cmd.run()
    if ret.runtime.returncode == 0:
        return xml.dom.minidom.parseString(ret.runtime.stdout)
    else:
        raise Exception(cmd.cmdline + ' failed:\n%s' % ret.runtime.stderr)