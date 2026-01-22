from typing import List, Optional
from ..exc import unimplemented
from .base import VariableTracker
from .constant import ConstantVariable
class CountIteratorVariable(IteratorVariable):

    def __init__(self, item: int=0, step: int=1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(item, VariableTracker):
            item = ConstantVariable.create(item)
        if not isinstance(step, VariableTracker):
            step = ConstantVariable.create(step)
        self.item = item
        self.step = step

    def next_variables(self, tx):
        assert self.mutable_local
        next_item = self.item.call_method(tx, '__add__', [self.step], {})
        next_iter = self.clone(item=next_item)
        tx.replace_all(self, next_iter)
        return (self.item, next_iter)