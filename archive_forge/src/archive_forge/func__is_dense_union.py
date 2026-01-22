from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _is_dense_union(type_: 'pyarrow.DataType') -> bool:
    """Whether the provided Arrow type is a dense union."""
    import pyarrow as pa
    return pa.types.is_union(type_) and type_.mode == 'dense'