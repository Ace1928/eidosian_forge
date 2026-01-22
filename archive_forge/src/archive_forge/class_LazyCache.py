import functools
from typing import Optional
from .base import VariableTracker
class LazyCache:
    """Container to cache the real VariableTracker"""

    def __init__(self, value, source):
        assert source
        self.value = value
        self.source = source
        self.vt: Optional[VariableTracker] = None

    def realize(self, parents_tracker):
        assert self.vt is None
        from ..symbolic_convert import InstructionTranslator
        from .builder import VariableBuilder
        tx = InstructionTranslator.current_tx()
        self.vt = VariableBuilder(tx, self.source)(self.value)
        self.vt.parents_tracker.add(parents_tracker)
        del self.value
        del self.source