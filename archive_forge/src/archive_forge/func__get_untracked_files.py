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
def _get_untracked_files(self, *args: Any, **kwargs: Any) -> List[str]:
    proc = self.git.status(*args, porcelain=True, untracked_files=True, as_process=True, **kwargs)
    prefix = '?? '
    untracked_files = []
    for line in proc.stdout:
        line = line.decode(defenc)
        if not line.startswith(prefix):
            continue
        filename = line[len(prefix):].rstrip('\n')
        if filename[0] == filename[-1] == '"':
            filename = filename[1:-1]
            filename = filename.encode('ascii').decode('unicode_escape').encode('latin1').decode(defenc)
        untracked_files.append(filename)
    finalize_process(proc)
    return untracked_files