import logging
import os
import shutil
import sys
import urllib.parse
from typing import (
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import (
from pip._internal.utils.urls import get_url_scheme
@classmethod
def _is_local_repository(cls, repo: str) -> bool:
    """
        posix absolute paths start with os.path.sep,
        win32 ones start with drive (like c:\\folder)
        """
    drive, tail = os.path.splitdrive(repo)
    return repo.startswith(os.path.sep) or bool(drive)