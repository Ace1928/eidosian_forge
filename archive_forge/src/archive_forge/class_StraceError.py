import os
import signal
import subprocess
import tempfile
from . import errors
class StraceError(errors.BzrError):
    _fmt = 'strace failed: %(err_messages)s'