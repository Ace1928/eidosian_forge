import io
import itertools
import json
from dataclasses import dataclass
from typing import Optional
import pyarrow as pa
import pyarrow.json as paj
import datasets
from datasets.table import table_cast
from datasets.utils.file_utils import readline
@dataclass
class JsonConfig(datasets.BuilderConfig):
    """BuilderConfig for JSON."""
    features: Optional[datasets.Features] = None
    encoding: str = 'utf-8'
    encoding_errors: Optional[str] = None
    field: Optional[str] = None
    use_threads: bool = True
    block_size: Optional[int] = None
    chunksize: int = 10 << 20
    newlines_in_values: Optional[bool] = None