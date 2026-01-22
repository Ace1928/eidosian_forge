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
def _get_config_path(self, config_level: Lit_config_levels, git_dir: Optional[PathLike]=None) -> str:
    if git_dir is None:
        git_dir = self.git_dir
    if os.name == 'nt' and config_level == 'system':
        config_level = 'global'
    if config_level == 'system':
        return '/etc/gitconfig'
    elif config_level == 'user':
        config_home = os.environ.get('XDG_CONFIG_HOME') or osp.join(os.environ.get('HOME', '~'), '.config')
        return osp.normpath(osp.expanduser(osp.join(config_home, 'git', 'config')))
    elif config_level == 'global':
        return osp.normpath(osp.expanduser('~/.gitconfig'))
    elif config_level == 'repository':
        repo_dir = self._common_dir or git_dir
        if not repo_dir:
            raise NotADirectoryError
        else:
            return osp.normpath(osp.join(repo_dir, 'config'))
    else:
        assert_never(config_level, ValueError(f'Invalid configuration level: {config_level!r}'))