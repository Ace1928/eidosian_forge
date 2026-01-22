import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union, cast
import pgzip
import torch
from torch import Tensor
from fairscale.internal.containers import from_np, to_np
from .utils import ExitCode
def _copy_compressed(src: Path, dest: Path, thread: Optional[int], blocksize: int) -> Tuple[int, int]:
    """Helper to copy a file and compress it at the same time.

    Returns:
        (int, int):
            original size and compressed size in bytes.
    """
    with open(str(src), 'rb') as srcf:
        with pgzip.open(str(dest), 'wb', compresslevel=5, thread=thread, blocksize=blocksize) as destf:
            while True:
                buf = srcf.read(blocksize)
                if len(buf) == 0:
                    break
                destf.write(buf)
    orig, comp = (Path(src).stat().st_size, Path(dest).stat().st_size)
    assert orig >= comp or comp < 1 * 1024 * 1024, f'Compressed size {comp} > original {orig} for large data'
    return (orig, comp)