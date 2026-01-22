import inspect
from typing import Dict, List
import torch
from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable
class PlacementVariable(DistributedVariable):

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_placement(value):
        if not DistributedVariable.is_available():
            return False
        from torch.distributed._tensor.placement_types import Placement
        return isinstance(value, Placement)

    def as_python_constant(self):
        return self.value

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from . import ConstantVariable
        allowed_methods = ['__init__', '__setattr__']
        if name in allowed_methods:
            try:
                value_type = type(self.value)
                assert inspect.getattr_static(value_type, '__getattr__', None) is None, 'no custom getattr allowed!'
                method = inspect.getattr_static(value_type, name)
            except AttributeError:
                method = None
            if method is object.__init__:
                return ConstantVariable.create(None)
            args = [x.as_python_constant() for x in args]
            kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
            method(self.value, *args, **kwargs)
            return self
        return super().call_method(tx, name, args, kwargs)