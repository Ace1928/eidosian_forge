from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def abort_on_errors(node):
    if Errors.get_errors_count() != 0:
        raise AbortError('pipeline break')
    return node