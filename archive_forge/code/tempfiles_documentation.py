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
Release this context

        This releases the current context, potentially deleting all
        managed temporary objects (files and directories), and resetting
        the context to generate unique names.

        Parameters
        ----------
        remove: bool
            If ``True``, delete all managed files / directories
        