import os
import re
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional, Union
from zipfile import ZipFile
from ..utils.file_utils import cached_path, hf_github_url
from ..utils.logging import get_logger
from ..utils.version import Version
def _iter_archive_members(path):
    dummy_parent_path = Path(self.dummy_file).parent
    relative_path = path.relative_to(dummy_parent_path)
    with ZipFile(self.local_path_to_dummy_data) as zip_file:
        members = zip_file.namelist()
    for member in members:
        if member.startswith(relative_path.as_posix()):
            yield dummy_parent_path.joinpath(member)