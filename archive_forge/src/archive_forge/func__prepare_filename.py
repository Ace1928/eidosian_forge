import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TextIO, Union
from lightning_fabric.utilities.cloud_io import get_filesystem
def _prepare_filename(self, action_name: Optional[str]=None, extension: str='.txt', split_token: str='-') -> str:
    args = []
    if self._stage is not None:
        args.append(self._stage)
    if self.filename:
        args.append(self.filename)
    if self._local_rank is not None:
        args.append(str(self._local_rank))
    if action_name is not None:
        args.append(action_name)
    return split_token.join(args) + extension