import time
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema
def _convert_dataclass_list(self, key, cls):
    values = getattr(self, key)
    if not isinstance(values, list):
        raise ValueError(f'`{key}` must be a list')
    if len(values) > 0:
        if all((isinstance(v, dict) for v in values)):
            try:
                setattr(self, key, [cls(**v) for v in values])
            except TypeError as e:
                raise ValueError(f'Error when coercing {values} to {cls.__name__}: {e}')
        elif any((not isinstance(v, cls) for v in values)):
            raise ValueError(f'Items in `{key}` must all have the same type: {cls.__name__} or dict')