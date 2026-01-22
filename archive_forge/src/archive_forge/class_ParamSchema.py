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
@experimental
class ParamSchema:
    """
    Specification of parameters applicable to the model.
    ParamSchema is represented as a list of :py:class:`ParamSpec`.
    """

    def __init__(self, params: List[ParamSpec]):
        if not all((isinstance(x, ParamSpec) for x in params)):
            raise MlflowException.invalid_parameter_value(f'ParamSchema inputs only accept {ParamSchema.__class__}')
        if (duplicates := self._find_duplicates(params)):
            raise MlflowException.invalid_parameter_value(f'Duplicated parameters found in schema: {duplicates}')
        self._params = params

    @staticmethod
    def _find_duplicates(params: List[ParamSpec]) -> List[str]:
        param_names = [param_spec.name for param_spec in params]
        uniq_param = set()
        duplicates = []
        for name in param_names:
            if name in uniq_param:
                duplicates.append(name)
            else:
                uniq_param.add(name)
        return duplicates

    def __len__(self):
        return len(self._params)

    def __iter__(self):
        return iter(self._params)

    @property
    def params(self) -> List[ParamSpec]:
        """Representation of ParamSchema as a list of ParamSpec."""
        return self._params

    def to_json(self) -> str:
        """Serialize into json string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from a json string."""
        return cls([ParamSpec.from_json_dict(**x) for x in json.loads(json_str)])

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize into a jsonable dictionary."""
        return [x.to_dict() for x in self.params]

    def __eq__(self, other) -> bool:
        if isinstance(other, ParamSchema):
            return self.params == other.params
        return False

    def __repr__(self) -> str:
        return repr(self.params)