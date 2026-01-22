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
def _rel_to_abs_instr(rel_instr, name2len):
    """Returns _AbsoluteInstruction instance for given RelativeInstruction.

    Args:
        rel_instr: RelativeInstruction instance.
        name2len: dict {split_name: num_examples}.
    """
    pct_to_abs = _pct_to_abs_closest if rel_instr.rounding == 'closest' else _pct_to_abs_pct1
    split = rel_instr.splitname
    if split not in name2len:
        raise ValueError(f'Unknown split "{split}". Should be one of {list(name2len)}.')
    num_examples = name2len[split]
    from_ = rel_instr.from_
    to = rel_instr.to
    if rel_instr.unit == '%':
        from_ = 0 if from_ is None else pct_to_abs(from_, num_examples)
        to = num_examples if to is None else pct_to_abs(to, num_examples)
    else:
        from_ = 0 if from_ is None else from_
        to = num_examples if to is None else to
    if from_ < 0:
        from_ = max(num_examples + from_, 0)
    if to < 0:
        to = max(num_examples + to, 0)
    from_ = min(from_, num_examples)
    to = min(to, num_examples)
    return _AbsoluteInstruction(split, from_, to)