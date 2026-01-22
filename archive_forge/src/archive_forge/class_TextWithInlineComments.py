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
class TextWithInlineComments(Base):
    text: str
    _inline_comments: Optional[LList[internal.InlineComment]] = Field(default_factory=lambda: None, repr=False)