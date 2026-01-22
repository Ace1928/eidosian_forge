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
def _validate_pass(self, pass_cls):
    if not (isinstance(pass_cls, str) or (inspect.isclass(pass_cls) and issubclass(pass_cls, CompilerPass))):
        msg = 'Pass must be referenced by name or be a subclass of a CompilerPass. Have %s' % pass_cls
        raise TypeError(msg)
    if isinstance(pass_cls, str):
        pass_cls = _pass_registry.find_by_name(pass_cls)
    elif not _pass_registry.is_registered(pass_cls):
        raise ValueError('Pass %s is not registered' % pass_cls)