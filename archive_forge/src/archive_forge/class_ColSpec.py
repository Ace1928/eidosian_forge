import builtins
import datetime as dt
import importlib.util
import json
import string
import warnings
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
class ColSpec:
    """
    Specification of name and type of a single column in a dataset.
    """

    def __init__(self, type: Union[DataType, Array, Object, str], name: Optional[str]=None, optional: Optional[bool]=None, required: Optional[bool]=None):
        self._name = name
        if optional is not None:
            if required is not None:
                raise MlflowException('Only one of `optional` and `required` can be specified. `optional` is deprecated, please use `required` instead.')
            else:
                warnings.warn('`optional` is deprecated and will be removed in a future version of MLflow. Use `required` instead.', category=FutureWarning)
                self._required = not optional
        else:
            self._required = True if required is None else required
        try:
            self._type = DataType[type] if isinstance(type, str) else type
        except KeyError:
            raise MlflowException(f"Unsupported type '{type}', expected instance of DataType or one of {[t.name for t in DataType]}")
        if not isinstance(self.type, (DataType, Array, Object, Map)):
            raise TypeError(f"Expected mlflow.types.schema.Datatype, mlflow.types.schema.Array, mlflow.types.schema.Object, mlflow.types.schema.Map or str for the 'type' argument, but got {self.type.__class__}")

    @property
    def type(self) -> Union[DataType, Array, Object]:
        """The column data type."""
        return self._type

    @property
    def name(self) -> Optional[str]:
        """The column name or None if the columns is unnamed."""
        return self._name

    @experimental
    @property
    def optional(self) -> bool:
        """
        Whether this column is optional.

        .. Warning:: Deprecated. `optional` is deprecated in favor of `required`.
        """
        return not self._required

    @experimental
    @property
    def required(self) -> bool:
        """Whether this column is required."""
        return self._required

    def to_dict(self) -> Dict[str, Any]:
        d = {'type': self.type.name} if isinstance(self.type, DataType) else self.type.to_dict()
        if self.name is not None:
            d['name'] = self.name
        d['required'] = self.required
        return d

    def __eq__(self, other) -> bool:
        if isinstance(other, ColSpec):
            names_eq = self.name is None and other.name is None or self.name == other.name
            return names_eq and self.type == other.type and (self.required == other.required)
        return False

    def __repr__(self) -> str:
        required = 'required' if self.required else 'optional'
        if self.name is None:
            return f'{self.type!r} ({required})'
        return f'{self.name!r}: {self.type!r} ({required})'

    @classmethod
    def from_json_dict(cls, **kwargs):
        """
        Deserialize from a json loaded dictionary.
        The dictionary is expected to contain `type` and
        optional `name` and `required` keys.
        """
        if not {'type'} <= set(kwargs.keys()):
            raise MlflowException('Missing keys in ColSpec JSON. Expected to find key `type`')
        if kwargs['type'] not in [ARRAY_TYPE, OBJECT_TYPE, MAP_TYPE]:
            return cls(**kwargs)
        name = kwargs.pop('name', None)
        optional = kwargs.pop('optional', None)
        required = kwargs.pop('required', None)
        if kwargs['type'] == ARRAY_TYPE:
            return cls(name=name, type=Array.from_json_dict(**kwargs), optional=optional, required=required)
        if kwargs['type'] == OBJECT_TYPE:
            return cls(name=name, type=Object.from_json_dict(**kwargs), optional=optional, required=required)
        if kwargs['type'] == MAP_TYPE:
            return cls(name=name, type=Map.from_json_dict(**kwargs), optional=optional, required=required)