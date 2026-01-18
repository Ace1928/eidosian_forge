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
def write_cache(entries: Sequence[Union[BaseIndexEntry, 'IndexEntry']], stream: IO[bytes], extension_data: Union[None, bytes]=None, ShaStreamCls: Type[IndexFileSHA1Writer]=IndexFileSHA1Writer) -> None:
    """Write the cache represented by entries to a stream.

    :param entries: **sorted** list of entries

    :param stream: stream to wrap into the AdapterStreamCls - it is used for
        final output.

    :param ShaStreamCls: Type to use when writing to the stream. It produces a sha
        while writing to it, before the data is passed on to the wrapped stream

    :param extension_data: any kind of data to write as a trailer, it must begin
        a 4 byte identifier, followed by its size (4 bytes).
    """
    stream_sha = ShaStreamCls(stream)
    tell = stream_sha.tell
    write = stream_sha.write
    version = 2
    write(b'DIRC')
    write(pack('>LL', version, len(entries)))
    for entry in entries:
        beginoffset = tell()
        write(entry.ctime_bytes)
        write(entry.mtime_bytes)
        path_str = str(entry.path)
        path: bytes = force_bytes(path_str, encoding=defenc)
        plen = len(path) & CE_NAMEMASK
        assert plen == len(path), 'Path %s too long to fit into index' % entry.path
        flags = plen | entry.flags & CE_NAMEMASK_INV
        write(pack('>LLLLLL20sH', entry.dev, entry.inode, entry.mode, entry.uid, entry.gid, entry.size, entry.binsha, flags))
        write(path)
        real_size = tell() - beginoffset + 8 & ~7
        write(b'\x00' * (beginoffset + real_size - tell()))
    if extension_data is not None:
        stream_sha.write(extension_data)
    stream_sha.write_sha()