import os
import shutil
import time
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
from ..utils import (
def test_unique_name_long():
    """The file name should never be longer than 255 characters"""
    url = f'https://www.something.com/data{'a' * 500}.txt'
    assert len(url) > 255
    fname = unique_file_name(url)
    assert len(fname) == 255
    assert fname[-10:] == 'aaaaaa.txt'
    assert fname.split('-')[1][:10] == 'aaaaaaaaaa'