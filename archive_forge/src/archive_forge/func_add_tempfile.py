import os
import time
import tempfile
import logging
import shutil
import weakref
from pyomo.common.dependencies import attempt_import, pyutilib_available
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TempfileContextError
from pyomo.common.multithread import MultiThreadWrapperWithMain
def add_tempfile(self, filename, exists=True):
    """Declare the specified file/directory to be temporary.

        This adds the specified path as a "temporary" object to this
        context's list of managed temporary paths (i.e., it will be
        potentially be deleted when the context is released (see
        :meth:`release`).

        Parameters
        ----------
        filename: str
            the file / directory name to be treated as temporary
        exists: bool
            if ``True``, the file / directory must already exist.

        """
    tmp = os.path.abspath(filename)
    if exists and (not os.path.exists(tmp)):
        raise IOError('Temporary file does not exist: ' + tmp)
    self.tempfiles.append((None, tmp))