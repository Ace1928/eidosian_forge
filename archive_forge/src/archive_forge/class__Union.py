from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
class _Union:

    @classmethod
    def create(cls, **kwargs):
        assert len(kwargs) == 1
        return cls(**{**{f.name: None for f in fields(cls)}, **kwargs})

    def __post_init__(self):
        assert sum((1 for f in fields(self) if getattr(self, f.name) is not None)) == 1

    @property
    def value(self):
        val = next((getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None), None)
        assert val is not None
        return val

    @property
    def type(self):
        val_type = next((f.name for f in fields(self) if getattr(self, f.name) is not None), None)
        assert val_type is not None
        return val_type

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{type(self).__name__}({self.type}={self.value})'