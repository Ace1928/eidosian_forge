import os
import shutil
import time
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
from ..utils import (
def test_temporary_file_path():
    """Make sure the file is writable and cleaned up in the end when given a dir"""
    with TemporaryDirectory() as path:
        with temporary_file(path) as tmp:
            assert Path(tmp).exists()
            assert path in tmp
            with open(tmp, 'w', encoding='utf-8') as outfile:
                outfile.write('Meh')
            with open(tmp, encoding='utf-8') as infile:
                assert infile.read().strip() == 'Meh'
        assert not Path(tmp).exists()