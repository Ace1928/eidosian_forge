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
class ArrowReader(BaseReader):
    """
    Build a Dataset object out of Instruction instance(s).
    This Reader uses either memory mapping or file descriptors (in-memory) on arrow files.
    """

    def __init__(self, path: str, info: Optional['DatasetInfo']):
        """Initializes ArrowReader.

        Args:
            path (str): path where Arrow files are stored.
            info (DatasetInfo): info about the dataset.
        """
        super().__init__(path, info)
        self._filetype_suffix = 'arrow'

    def _get_table_from_filename(self, filename_skip_take, in_memory=False) -> Table:
        """Returns a Dataset instance from given (filename, skip, take)."""
        filename, skip, take = (filename_skip_take['filename'], filename_skip_take['skip'] if 'skip' in filename_skip_take else None, filename_skip_take['take'] if 'take' in filename_skip_take else None)
        table = ArrowReader.read_table(filename, in_memory=in_memory)
        if take == -1:
            take = len(table) - skip
        if skip is not None and take is not None and (not (skip == 0 and take == len(table))):
            table = table.slice(skip, take)
        return table

    @staticmethod
    def read_table(filename, in_memory=False) -> Table:
        """
        Read table from file.

        Args:
            filename (str): File name of the table.
            in_memory (bool, default=False): Whether to copy the data in-memory.

        Returns:
            pyarrow.Table
        """
        table_cls = InMemoryTable if in_memory else MemoryMappedTable
        return table_cls.from_file(filename)