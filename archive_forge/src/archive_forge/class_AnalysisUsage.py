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
class AnalysisUsage(object):
    """This looks and behaves like LLVM's AnalysisUsage because its like that.
    """

    def __init__(self):
        self._required = set()
        self._preserved = set()

    def get_required_set(self):
        return self._required

    def get_preserved_set(self):
        return self._preserved

    def add_required(self, pss):
        self._required.add(pss)

    def add_preserved(self, pss):
        self._preserved.add(pss)

    def __str__(self):
        return 'required: %s\n' % self._required