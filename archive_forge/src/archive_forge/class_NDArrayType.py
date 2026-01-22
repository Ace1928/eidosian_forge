import datetime
import math
import typing as t
from wandb.util import (
class NDArrayType(Type):
    """Represents a list of homogenous types."""
    name = 'ndarray'
    types: t.ClassVar[t.List[type]] = []
    _serialization_path: t.Optional[t.Dict[str, str]]

    def __init__(self, shape: t.Sequence[int], serialization_path: t.Optional[t.Dict[str, str]]=None):
        self.params.update({'shape': list(shape)})
        self._serialization_path = serialization_path

    @classmethod
    def from_obj(cls, py_obj: t.Optional[t.Any]=None) -> 'NDArrayType':
        if is_numpy_array(py_obj):
            return cls(py_obj.shape)
        elif isinstance(py_obj, list):
            shape = []
            target = py_obj
            while isinstance(target, list):
                dim = len(target)
                shape.append(dim)
                if dim > 0:
                    target = target[0]
            return cls(shape)
        else:
            raise TypeError('NDArrayType.from_obj expects py_obj to be ndarray or list, found {}'.format(py_obj.__class__))

    def assign_type(self, wb_type: 'Type') -> t.Union['NDArrayType', InvalidType]:
        if isinstance(wb_type, NDArrayType) and self.params['shape'] == wb_type.params['shape']:
            return self
        elif isinstance(wb_type, ListType):
            return self
        return InvalidType()

    def assign(self, py_obj: t.Optional[t.Any]=None) -> t.Union['NDArrayType', InvalidType]:
        if is_numpy_array(py_obj) or isinstance(py_obj, list):
            py_type = self.from_obj(py_obj)
            return self.assign_type(py_type)
        return InvalidType()

    def to_json(self, artifact: t.Optional['Artifact']=None) -> t.Dict[str, t.Any]:
        res = {'wb_type': self.name, 'params': {'shape': self.params['shape'], 'serialization_path': self._serialization_path}}
        return res

    def _get_serialization_path(self) -> t.Optional[t.Dict[str, str]]:
        return self._serialization_path

    def _set_serialization_path(self, path: str, key: str) -> None:
        self._serialization_path = {'path': path, 'key': key}

    def _clear_serialization_path(self) -> None:
        self._serialization_path = None