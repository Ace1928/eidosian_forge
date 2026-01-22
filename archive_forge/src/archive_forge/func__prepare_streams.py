import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TextIO, Union
from lightning_fabric.utilities.cloud_io import get_filesystem
def _prepare_streams(self) -> None:
    if self._write_stream is not None:
        return
    if self.filename and self.dirpath:
        filepath = os.path.join(self.dirpath, self._prepare_filename())
        fs = get_filesystem(filepath)
        fs.mkdirs(self.dirpath, exist_ok=True)
        file = fs.open(filepath, 'a')
        self._output_file = file
        self._write_stream = file.write
    else:
        self._write_stream = self._rank_zero_info