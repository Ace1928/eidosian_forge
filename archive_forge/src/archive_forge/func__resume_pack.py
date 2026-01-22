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
def _resume_pack(self, name):
    """Get a suspended Pack object by name.

        :param name: The name of the pack - e.g. '123456'
        :return: A Pack object.
        """
    if not re.match('[a-f0-9]{32}', name):
        raise errors.UnresumableWriteGroup(self.repo, [name], 'Malformed write group token')
    try:
        rev_index = self._make_index(name, '.rix', resume=True)
        inv_index = self._make_index(name, '.iix', resume=True)
        txt_index = self._make_index(name, '.tix', resume=True)
        sig_index = self._make_index(name, '.six', resume=True)
        if self.chk_index is not None:
            chk_index = self._make_index(name, '.cix', resume=True, is_chk=True)
        else:
            chk_index = None
        result = self.resumed_pack_factory(name, rev_index, inv_index, txt_index, sig_index, self._upload_transport, self._pack_transport, self._index_transport, self, chk_index=chk_index)
    except _mod_transport.NoSuchFile as e:
        raise errors.UnresumableWriteGroup(self.repo, [name], str(e))
    self.add_pack_to_memory(result)
    self._resumed_packs.append(result)
    return result