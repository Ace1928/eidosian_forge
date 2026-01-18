import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
def test_tar_no_error_on_close():
    with io.BytesIO() as buffer:
        with icom._BytesTarFile(fileobj=buffer, mode='w'):
            pass