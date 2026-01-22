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
def _make_script(self, entry, filenames, options=None):
    post_interp = b''
    if options:
        args = options.get('interpreter_args', [])
        if args:
            args = ' %s' % ' '.join(args)
            post_interp = args.encode('utf-8')
    shebang = self._get_shebang('utf-8', post_interp, options=options)
    script = self._get_script_text(entry).encode('utf-8')
    scriptnames = self.get_script_filenames(entry.name)
    if options and options.get('gui', False):
        ext = 'pyw'
    else:
        ext = 'py'
    self._write_script(scriptnames, shebang, script, filenames, ext)