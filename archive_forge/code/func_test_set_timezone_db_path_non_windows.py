import os
import subprocess
import sys
import pytest
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
@pytest.mark.skipif(sys.platform == 'win32', reason='Path to timezone database is not configurable on non-Windows platforms')
def test_set_timezone_db_path_non_windows():
    with pytest.raises(ArrowInvalid, match='Arrow was set to use OS timezone database at compile time'):
        pa.set_timezone_db_path('path')