import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
def _config_parser_constrained(self, read_only: bool) -> SectionConstraint:
    """:return: Config Parser constrained to our submodule in read or write mode"""
    try:
        pc: Union['Commit_ish', None] = self.parent_commit
    except ValueError:
        pc = None
    parser = self._config_parser(self.repo, pc, read_only)
    parser.set_submodule(self)
    return SectionConstraint(parser, sm_section(self.name))