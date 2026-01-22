from __future__ import annotations
import gc
import logging
import os
import os.path as osp
from pathlib import Path
import re
import shlex
import warnings
import gitdb
from gitdb.db.loose import LooseObjectDB
from gitdb.exc import BadObject
from git.cmd import Git, handle_process_output
from git.compat import defenc, safe_decode
from git.config import GitConfigParser
from git.db import GitCmdObjectDB
from git.exc import (
from git.index import IndexFile
from git.objects import Submodule, RootModule, Commit
from git.refs import HEAD, Head, Reference, TagReference
from git.remote import Remote, add_progress, to_progress_instance
from git.util import (
from .fun import (
from git.types import (
from typing import (
from git.types import ConfigLevels_Tup, TypedDict
def _get_alternates(self) -> List[str]:
    """The list of alternates for this repo from which objects can be retrieved.

        :return: List of strings being pathnames of alternates
        """
    if self.git_dir:
        alternates_path = osp.join(self.git_dir, 'objects', 'info', 'alternates')
    if osp.exists(alternates_path):
        with open(alternates_path, 'rb') as f:
            alts = f.read().decode(defenc)
        return alts.strip().splitlines()
    return []