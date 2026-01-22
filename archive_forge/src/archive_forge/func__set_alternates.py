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
def _set_alternates(self, alts: List[str]) -> None:
    """Sets the alternates.

        :param alts:
            is the array of string paths representing the alternates at which
            git should look for objects, i.e. /home/user/repo/.git/objects

        :raise NoSuchPathError:

        :note:
            The method does not check for the existence of the paths in alts
            as the caller is responsible.
        """
    alternates_path = osp.join(self.common_dir, 'objects', 'info', 'alternates')
    if not alts:
        if osp.isfile(alternates_path):
            os.remove(alternates_path)
    else:
        with open(alternates_path, 'wb') as f:
            f.write('\n'.join(alts).encode(defenc))