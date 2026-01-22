import datetime
import math
import typing as t
from wandb.util import (
class PythonObjectType(Type):
    """A backup type that keeps track of the python object name."""
    name = 'pythonObject'
    legacy_names = ['object']
    types: t.ClassVar[t.List[type]] = []

    def __init__(self, class_name: str):
        self.params.update({'class_name': class_name})

    @classmethod
    def from_obj(cls, py_obj: t.Optional[t.Any]=None) -> 'PythonObjectType':
        return cls(py_obj.__class__.__name__)