from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
class ByteCodePy311(_ByteCode):

    def __init__(self, func_id):
        super().__init__(func_id)
        entries = dis.Bytecode(func_id.code).exception_entries
        self.exception_entries = tuple(map(self.fixup_eh, entries))

    @staticmethod
    def fixup_eh(ent):
        out = dis._ExceptionTableEntry(start=ent.start + _FIXED_OFFSET, end=ent.end + _FIXED_OFFSET, target=ent.target + _FIXED_OFFSET, depth=ent.depth, lasti=ent.lasti)
        return out

    def find_exception_entry(self, offset):
        """
        Returns the exception entry for the given instruction offset
        """
        candidates = []
        for ent in self.exception_entries:
            if ent.start <= offset < ent.end:
                candidates.append((ent.depth, ent))
        if candidates:
            ent = max(candidates)[1]
            return ent