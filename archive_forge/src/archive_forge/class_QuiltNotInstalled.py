import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
class QuiltNotInstalled(errors.BzrError):
    _fmt = 'Quilt is not installed.'