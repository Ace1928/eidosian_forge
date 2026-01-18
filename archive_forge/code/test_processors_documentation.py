from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import pytest
from .. import Pooch
from ..processors import Unzip, Untar, Decompress
from .utils import pooch_test_url, pooch_test_registry, check_tiny_data, capture_log

    Test that calling with invalid members then without them works.
    https://github.com/fatiando/pooch/issues/364
    