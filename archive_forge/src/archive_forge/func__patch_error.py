import timeit
from abc import abstractmethod, ABCMeta
from collections import namedtuple, OrderedDict
import inspect
from pprint import pformat
from numba.core.compiler_lock import global_compiler_lock
from numba.core import errors, config, transforms, utils
from numba.core.tracing import event
from numba.core.postproc import PostProcessor
from numba.core.ir_utils import enforce_no_dels, legalize_single_scope
import numba.core.event as ev
def _patch_error(self, desc, exc):
    """
        Patches the error to show the stage that it arose in.
        """
    newmsg = '{desc}\n{exc}'.format(desc=desc, exc=exc)
    exc.args = (newmsg,)
    return exc