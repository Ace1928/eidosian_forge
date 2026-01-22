import os
import io
import json
import pandas as pd
import pyarrow
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import ray
from ray.air.constants import (
from ray.util.annotations import PublicAPI
import logging
@staticmethod
def _read_file_as_str(storage_filesystem: pyarrow.fs.FileSystem, storage_path: str) -> str:
    """Opens a file as an input stream reading all byte content sequentially and
         decoding read bytes as utf-8 string.

        Args:
            storage_filesystem: The filesystem to use.
            storage_path: The source to open for reading.
        """
    with storage_filesystem.open_input_stream(storage_path) as f:
        return f.readall().decode()