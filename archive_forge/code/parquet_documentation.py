import itertools
from dataclasses import dataclass
from typing import List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import datasets
from datasets.table import table_cast
We handle string, list and dicts in datafiles