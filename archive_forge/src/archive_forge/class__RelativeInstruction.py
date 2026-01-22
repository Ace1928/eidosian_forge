import copy
import math
import os
import re
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.contrib.concurrent import thread_map
from .download.download_config import DownloadConfig
from .naming import _split_re, filenames_for_dataset_split
from .table import InMemoryTable, MemoryMappedTable, Table, concat_tables
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import cached_path
@dataclass(frozen=True)
class _RelativeInstruction:
    """Represents a single parsed slicing instruction, can use % and negatives."""
    splitname: str
    from_: Optional[int] = None
    to: Optional[int] = None
    unit: Optional[str] = None
    rounding: Optional[str] = None

    def __post_init__(self):
        if self.unit is not None and self.unit not in ['%', 'abs']:
            raise ValueError('unit must be either % or abs')
        if self.rounding is not None and self.rounding not in ['closest', 'pct1_dropremainder']:
            raise ValueError('rounding must be either closest or pct1_dropremainder')
        if self.unit != '%' and self.rounding is not None:
            raise ValueError('It is forbidden to specify rounding if not using percent slicing.')
        if self.unit == '%' and self.from_ is not None and (abs(self.from_) > 100):
            raise ValueError('Percent slice boundaries must be > -100 and < 100.')
        if self.unit == '%' and self.to is not None and (abs(self.to) > 100):
            raise ValueError('Percent slice boundaries must be > -100 and < 100.')
        self.__dict__['rounding'] = 'closest' if self.rounding is None and self.unit == '%' else self.rounding