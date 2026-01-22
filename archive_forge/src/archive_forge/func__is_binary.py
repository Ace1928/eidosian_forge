from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _is_binary(type_: 'pyarrow.DataType') -> bool:
    """Whether the provided Array type is a variable-sized binary type."""
    import pyarrow as pa
    return pa.types.is_string(type_) or pa.types.is_large_string(type_) or pa.types.is_binary(type_) or pa.types.is_large_binary(type_)