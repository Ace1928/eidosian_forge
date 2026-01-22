import os
import shutil
import tempfile
import warnings
from functools import partial
from importlib import resources
from pathlib import Path
from pickle import dumps, loads
import numpy as np
import pytest
from sklearn.datasets import (
from sklearn.datasets._base import (
from sklearn.datasets.tests.test_common import check_as_frame
from sklearn.preprocessing import scale
from sklearn.utils import Bunch
class _DummyPath:
    """Minimal class that implements the os.PathLike interface."""

    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        return self.path