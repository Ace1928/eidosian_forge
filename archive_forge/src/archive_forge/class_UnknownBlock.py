import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
@dataclass(config=ConfigDict(validate_assignment=True, extra='allow', slots=True))
class UnknownBlock(Block):

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        attributes = ', '.join((f'{key}={value!r}' for key, value in self.__dict__.items()))
        return f'{class_name}({attributes})'

    def to_model(self):
        d = self.__dict__
        return internal.UnknownBlock.model_validate(d)

    @classmethod
    def from_model(cls, model: internal.UnknownBlock):
        d = model.model_dump()
        return cls(**d)