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
class OrderedListItem(Base):
    text: TextLikeField = ''

    def to_model(self):
        return internal.ListItem(children=[internal.Paragraph(children=_text_to_internal_children(self.text))], ordered=True)