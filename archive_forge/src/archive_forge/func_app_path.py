from __future__ import absolute_import
import pathlib
import contextlib
from typing import Optional, Union, Dict, TYPE_CHECKING
from lazyops.types.models import BaseSettings
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
from lazyops.imports._aiokeydb import (
@lazyproperty
def app_path(self) -> FileLike:
    app_dir = self.app_dir or self.context_path.joinpath('app')
    return File(app_dir) if _fileio_available else pathlib.Path(app_dir)