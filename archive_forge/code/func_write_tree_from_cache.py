from io import BytesIO
import os
import os.path as osp
from pathlib import Path
from stat import (
import subprocess
from git.cmd import handle_process_output, safer_popen
from git.compat import defenc, force_bytes, force_text, safe_decode
from git.exc import HookExecutionError, UnmergedEntriesError
from git.objects.fun import (
from git.util import IndexFileSHA1Writer, finalize_process
from gitdb.base import IStream
from gitdb.typ import str_tree_type
from .typ import BaseIndexEntry, IndexEntry, CE_NAMEMASK, CE_STAGESHIFT
from .util import pack, unpack
from typing import Dict, IO, List, Sequence, TYPE_CHECKING, Tuple, Type, Union, cast
from git.types import PathLike
def write_tree_from_cache(entries: List[IndexEntry], odb: 'GitCmdObjectDB', sl: slice, si: int=0) -> Tuple[bytes, List['TreeCacheTup']]:
    """Create a tree from the given sorted list of entries and put the respective
    trees into the given object database.

    :param entries: **Sorted** list of IndexEntries
    :param odb: Object database to store the trees in
    :param si: Start index at which we should start creating subtrees
    :param sl: Slice indicating the range we should process on the entries list
    :return: tuple(binsha, list(tree_entry, ...)) a tuple of a sha and a list of
        tree entries being a tuple of hexsha, mode, name
    """
    tree_items: List['TreeCacheTup'] = []
    ci = sl.start
    end = sl.stop
    while ci < end:
        entry = entries[ci]
        if entry.stage != 0:
            raise UnmergedEntriesError(entry)
        ci += 1
        rbound = entry.path.find('/', si)
        if rbound == -1:
            tree_items.append((entry.binsha, entry.mode, entry.path[si:]))
        else:
            base = entry.path[si:rbound]
            xi = ci
            while xi < end:
                oentry = entries[xi]
                orbound = oentry.path.find('/', si)
                if orbound == -1 or oentry.path[si:orbound] != base:
                    break
                xi += 1
            sha, _tree_entry_list = write_tree_from_cache(entries, odb, slice(ci - 1, xi), rbound + 1)
            tree_items.append((sha, S_IFDIR, base))
            ci = xi
    sio = BytesIO()
    tree_to_stream(tree_items, sio.write)
    sio.seek(0)
    istream = odb.store(IStream(str_tree_type, len(sio.getvalue()), sio))
    return (istream.binsha, tree_items)