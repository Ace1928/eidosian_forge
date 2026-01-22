import os
import posixpath
import sys
from io import BytesIO
from dulwich.errors import NoIndexPresent
from dulwich.file import FileLocked, _GitFile
from dulwich.object_store import (PACK_MODE, PACKDIR, PackBasedObjectStore,
from dulwich.objects import ShaFile
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, MemoryPackIndex, Pack,
from dulwich.refs import SymrefLoop
from dulwich.repo import (BASE_DIRECTORIES, COMMONDIR, CONTROLDIR,
from .. import osutils
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..errors import (AlreadyControlDirError, LockBroken, LockContention,
from ..lock import LogicalLockResult
from ..trace import warning
from ..transport import FileExists, NoSuchFile
from ..transport.local import LocalTransport
def _remove_packed_ref(self, name):
    if self._packed_refs is None:
        return
    self._packed_refs = None
    self.get_packed_refs()
    if name not in self._packed_refs:
        return
    del self._packed_refs[name]
    if name in self._peeled_refs:
        del self._peeled_refs[name]
    with self.transport.open_write_stream('packed-refs') as f:
        write_packed_refs(f, self._packed_refs, self._peeled_refs)