import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
def _path2trans_id(self, path):
    """Look up the trans id associated with a path.

        :param path: path to look up, None when the path does not exist
        :return: trans_id
        """
    trans_id = self._path2trans_id_cache.get(path, object)
    if trans_id is not object:
        return trans_id
    segments = osutils.splitpath(path)
    cur_parent = self._transform.root
    for cur_segment in segments:
        for child in self._all_children(cur_parent):
            final_name = self._final_name_cache.get(child)
            if final_name is None:
                final_name = self._transform.final_name(child)
                self._final_name_cache[child] = final_name
            if final_name == cur_segment:
                cur_parent = child
                break
        else:
            self._path2trans_id_cache[path] = None
            return None
    self._path2trans_id_cache[path] = cur_parent
    return cur_parent