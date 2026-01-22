from typing import List, Optional
from ..exc import unimplemented
from .base import VariableTracker
from .constant import ConstantVariable
class IteratorVariable(VariableTracker):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def next_variables(self, tx):
        unimplemented('abstract method, must implement')