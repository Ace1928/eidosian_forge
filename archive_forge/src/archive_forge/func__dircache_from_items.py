import base64
import collections
import io
import itertools
import logging
import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING
import fsspec.core
from ..asyn import AsyncFileSystem
from ..callbacks import DEFAULT_CALLBACK
from ..core import filesystem, open, split_protocol
from ..utils import isfilelike, merge_offset_ranges, other_paths
def _dircache_from_items(self):
    self.dircache = {'': []}
    it = self.references.items()
    for path, part in it:
        if isinstance(part, (bytes, str)):
            size = len(part)
        elif len(part) == 1:
            size = None
        else:
            _, _, size = part
        par = path.rsplit('/', 1)[0] if '/' in path else ''
        par0 = par
        subdirs = [par0]
        while par0 and par0 not in self.dircache:
            par0 = self._parent(par0)
            subdirs.append(par0)
        subdirs.reverse()
        for parent, child in zip(subdirs, subdirs[1:]):
            assert child not in self.dircache
            assert parent in self.dircache
            self.dircache[parent].append({'name': child, 'type': 'directory', 'size': 0})
            self.dircache[child] = []
        self.dircache[par].append({'name': path, 'type': 'file', 'size': size})