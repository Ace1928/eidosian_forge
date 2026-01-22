from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import pytest
from .. import Pooch
from ..processors import Unzip, Untar, Decompress
from .utils import pooch_test_url, pooch_test_registry, check_tiny_data, capture_log
def _check_logs(log_file, expected_lines):
    """
    Assert that the lines in the log match the expected ones.
    """
    lines = log_file.getvalue().splitlines()
    assert len(lines) == len(expected_lines)
    for line, expected_line in zip(lines, expected_lines):
        assert line.startswith(expected_line)