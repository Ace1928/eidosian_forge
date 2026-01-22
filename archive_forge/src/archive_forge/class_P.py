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
@dataclass(config=dataclass_config)
class P(Block):
    text: TextLikeField = ''

    def to_model(self):
        children = _text_to_internal_children(self.text)
        return internal.Paragraph(children=children)

    @classmethod
    def from_model(cls, model: internal.Paragraph):
        pieces = _internal_children_to_text(model.children)
        return cls(text=pieces)