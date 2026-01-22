import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
class RefRecorder(object):
    """
    An object which records events when instances created through it
    are deleted.  Custom events can also be recorded to aid in
    diagnosis.
    """

    def __init__(self):
        self._counts = collections.defaultdict(int)
        self._events = []
        self._wrs = {}

    def make_dummy(self, name):
        """
        Make an object whose deletion will be recorded as *name*.
        """
        return _Dummy(self, name)

    def _add_dummy(self, dummy):
        wr = weakref.ref(dummy, self._on_disposal)
        self._wrs[wr] = dummy.name
    __call__ = make_dummy

    def mark(self, event):
        """
        Manually append *event* to the recorded events.
        *event* can be formatted using format().
        """
        count = self._counts[event] + 1
        self._counts[event] = count
        self._events.append(event.format(count=count))

    def _on_disposal(self, wr):
        name = self._wrs.pop(wr)
        self._events.append(name)

    @property
    def alive(self):
        """
        A list of objects which haven't been deleted yet.
        """
        return [wr() for wr in self._wrs]

    @property
    def recorded(self):
        """
        A list of recorded events.
        """
        return self._events