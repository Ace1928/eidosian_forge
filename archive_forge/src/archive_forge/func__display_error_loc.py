import json
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union
from .json import pydantic_encoder
from .utils import Representation
def _display_error_loc(error: 'ErrorDict') -> str:
    return ' -> '.join((str(e) for e in error['loc']))