import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
class QuiltError(errors.BzrError):
    _fmt = 'An error (%(retcode)d) occurred running quilt: %(stderr)s%(extra)s'

    def __init__(self, retcode, stdout, stderr):
        self.retcode = retcode
        self.stderr = stderr
        if stdout is not None:
            self.extra = '\n\n%s' % stdout
        else:
            self.extra = ''
        self.stdout = stdout