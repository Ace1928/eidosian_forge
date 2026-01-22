import itertools
from dataclasses import dataclass
from typing import List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import datasets
from datasets.table import table_cast
@dataclass
class ParquetConfig(datasets.BuilderConfig):
    """BuilderConfig for Parquet."""
    batch_size: Optional[int] = None
    columns: Optional[List[str]] = None
    features: Optional[datasets.Features] = None