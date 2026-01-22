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
def _resolve_tempdir(self, dir=None):
    if dir is not None:
        return dir
    elif self.tempdir is not None:
        return self.tempdir
    elif self.manager().tempdir is not None:
        return self.manager().tempdir
    elif TempfileManager.main_thread.tempdir is not None:
        return TempfileManager.main_thread.tempdir
    elif pyutilib_available:
        if pyutilib_tempfiles.TempfileManager.tempdir is not None:
            deprecation_warning('The use of the PyUtilib TempfileManager.tempdir to specify the default location for Pyomo temporary files has been deprecated.  Please set TempfileManager.tempdir in pyomo.common.tempfiles', version='5.7.2')
            return pyutilib_tempfiles.TempfileManager.tempdir
    return None