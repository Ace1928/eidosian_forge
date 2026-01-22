from enum import Enum
from typing import List, Tuple, Union
from .tokenizers import (
from .implementations import (
class OffsetType(Enum):
    BYTE = 'byte'
    CHAR = 'char'