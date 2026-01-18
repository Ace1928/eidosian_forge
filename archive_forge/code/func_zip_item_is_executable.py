import logging
import os
import shutil
import stat
import tarfile
import zipfile
from typing import Iterable, List, Optional
from zipfile import ZipInfo
from pip._internal.exceptions import InstallationError
from pip._internal.utils.filetypes import (
from pip._internal.utils.misc import ensure_dir
def zip_item_is_executable(info: ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return bool(mode and stat.S_ISREG(mode) and mode & 73)