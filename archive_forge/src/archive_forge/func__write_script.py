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
def _write_script(self, names, shebang, script_bytes, filenames, ext):
    use_launcher = self.add_launchers and self._is_nt
    linesep = os.linesep.encode('utf-8')
    if not shebang.endswith(linesep):
        shebang += linesep
    if not use_launcher:
        script_bytes = shebang + script_bytes
    else:
        if ext == 'py':
            launcher = self._get_launcher('t')
        else:
            launcher = self._get_launcher('w')
        stream = BytesIO()
        with ZipFile(stream, 'w') as zf:
            source_date_epoch = os.environ.get('SOURCE_DATE_EPOCH')
            if source_date_epoch:
                date_time = time.gmtime(int(source_date_epoch))[:6]
                zinfo = ZipInfo(filename='__main__.py', date_time=date_time)
                zf.writestr(zinfo, script_bytes)
            else:
                zf.writestr('__main__.py', script_bytes)
        zip_data = stream.getvalue()
        script_bytes = launcher + shebang + zip_data
    for name in names:
        outname = os.path.join(self.target_dir, name)
        if use_launcher:
            n, e = os.path.splitext(outname)
            if e.startswith('.py'):
                outname = n
            outname = '%s.exe' % outname
            try:
                self._fileop.write_binary_file(outname, script_bytes)
            except Exception:
                logger.warning('Failed to write executable - trying to use .deleteme logic')
                dfname = '%s.deleteme' % outname
                if os.path.exists(dfname):
                    os.remove(dfname)
                os.rename(outname, dfname)
                self._fileop.write_binary_file(outname, script_bytes)
                logger.debug('Able to replace executable using .deleteme logic')
                try:
                    os.remove(dfname)
                except Exception:
                    pass
        else:
            if self._is_nt and (not outname.endswith('.' + ext)):
                outname = '%s.%s' % (outname, ext)
            if os.path.exists(outname) and (not self.clobber):
                logger.warning('Skipping existing file %s', outname)
                continue
            self._fileop.write_binary_file(outname, script_bytes)
            if self.set_mode:
                self._fileop.set_executable_mode([outname])
        filenames.append(outname)