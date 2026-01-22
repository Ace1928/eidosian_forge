import contextlib
import os
import shutil
import stat
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Optional, Union
import yaml
def _set_write_permission_and_retry(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)