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
def _get_script_text(self, entry):
    return self.script_template % dict(module=entry.prefix, import_name=entry.suffix.split('.')[0], func=entry.suffix)