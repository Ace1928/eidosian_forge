import os
import shutil
from pathlib import Path
import pytest
from .. import create, os_cache
from .. import __version__ as full_version
from .utils import check_tiny_data, capture_log
Fetch a data file from the local storage