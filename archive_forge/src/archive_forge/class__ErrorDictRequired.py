import json
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union
from .json import pydantic_encoder
from .utils import Representation
class _ErrorDictRequired(TypedDict):
    loc: Loc
    msg: str
    type: str