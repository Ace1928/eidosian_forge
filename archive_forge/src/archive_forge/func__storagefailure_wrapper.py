import contextlib
import errno
import io
import os
import shutil
import cachetools
import fasteners
from oslo_serialization import jsonutils
from oslo_utils import fileutils
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.utils import misc
@contextlib.contextmanager
def _storagefailure_wrapper():
    try:
        yield
    except exc.TaskFlowException:
        raise
    except Exception as e:
        if isinstance(e, (IOError, OSError)) and e.errno == errno.ENOENT:
            exc.raise_with_cause(exc.NotFound, 'Item not found: %s' % e.filename, cause=e)
        else:
            exc.raise_with_cause(exc.StorageFailure, 'Storage backend internal error', cause=e)