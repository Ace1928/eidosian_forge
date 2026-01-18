import os
import shutil
import time
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
from ..utils import (
def test_temporary_file_exception():
    """Make sure the file is writable and cleaned up when there is an exception"""
    try:
        with temporary_file() as tmp:
            assert Path(tmp).exists()
            raise ValueError('Nooooooooo!')
    except ValueError:
        assert not Path(tmp).exists()