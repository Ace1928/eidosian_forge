from io import BytesIO
import logging
import os
import re
import struct
import sys
import time
from zipfile import ZipInfo
from .compat import sysconfig, detect_encoding, ZipFile
from .resources import finder
from .util import (FileOperator, get_export_entry, convert_path,
import re
import sys
from %(module)s import %(import_name)s
def _get_launcher(self, kind):
    if struct.calcsize('P') == 8:
        bits = '64'
    else:
        bits = '32'
    platform_suffix = '-arm' if get_platform() == 'win-arm64' else ''
    name = '%s%s%s.exe' % (kind, bits, platform_suffix)
    distlib_package = __name__.rsplit('.', 1)[0]
    resource = finder(distlib_package).find(name)
    if not resource:
        msg = 'Unable to find resource %s in package %s' % (name, distlib_package)
        raise ValueError(msg)
    return resource.bytes