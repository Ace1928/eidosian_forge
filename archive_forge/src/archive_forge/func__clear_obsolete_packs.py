import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
def _clear_obsolete_packs(self, preserve=None):
    """Delete everything from the obsolete-packs directory.

        :return: A list of pack identifiers (the filename without '.pack') that
            were found in obsolete_packs.
        """
    found = []
    obsolete_pack_transport = self.transport.clone('obsolete_packs')
    if preserve is None:
        preserve = set()
    try:
        obsolete_pack_files = obsolete_pack_transport.list_dir('.')
    except _mod_transport.NoSuchFile:
        return found
    for filename in obsolete_pack_files:
        name, ext = osutils.splitext(filename)
        if ext == '.pack':
            found.append(name)
        if name in preserve:
            continue
        try:
            obsolete_pack_transport.delete(filename)
        except (errors.PathError, errors.TransportError) as e:
            warning("couldn't delete obsolete pack, skipping it:\n%s" % (e,))
    return found