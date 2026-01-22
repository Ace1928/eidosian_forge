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
def add_pass(self, pss, description=''):
    """
        Append a pass to the PassManager's compilation pipeline
        """
    self._validate_pass(pss)
    func_desc_tuple = (pss, description)
    self.passes.append(func_desc_tuple)
    self._finalized = False