from __future__ import annotations
import os
import stat
from pathlib import Path
from string import digits
from git.exc import WorkTreeRepositoryUnsupported
from git.objects import Object
from git.refs import SymbolicReference
from git.util import hex_to_bin, bin_to_hex, cygpath
from gitdb.exc import (
import os.path as osp
from git.cmd import Git
from typing import Union, Optional, cast, TYPE_CHECKING
from git.types import Commit_ish

    :return: Object at the given revision, either Commit, Tag, Tree or Blob
    :param rev: git-rev-parse compatible revision specification as string, please see
        http://www.kernel.org/pub/software/scm/git/docs/git-rev-parse.html
        for details
    :raise BadObject: if the given revision could not be found
    :raise ValueError: If rev couldn't be parsed
    :raise IndexError: If invalid reflog index is specified