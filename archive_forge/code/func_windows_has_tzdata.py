import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
def windows_has_tzdata():
    """
    This is the default location where tz.cpp will look for (until we make
    this configurable at run-time)
    """
    tzdata_bool = False
    if 'PYARROW_TZDATA_PATH' in os.environ:
        tzdata_bool = os.path.exists(os.environ['PYARROW_TZDATA_PATH'])
    if not tzdata_bool:
        tzdata_path = os.path.expandvars('%USERPROFILE%\\Downloads\\tzdata')
        tzdata_bool = os.path.exists(tzdata_path)
    return tzdata_bool