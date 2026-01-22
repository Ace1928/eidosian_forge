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
def _pct_to_abs_pct1(boundary, num_examples):
    if num_examples < 100:
        msg = 'Using "pct1_dropremainder" rounding on a split with less than 100 elements is forbidden: it always results in an empty dataset.'
        raise ValueError(msg)
    return boundary * math.trunc(num_examples / 100.0)