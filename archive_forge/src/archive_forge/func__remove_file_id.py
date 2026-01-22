from typing import Type
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import transport as _mod_transport
from ..repository import InterRepository, IsInWriteGroupError, Repository
from .repository import RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (InterSameDataRepository,
def _remove_file_id(self, file_id):
    t = self._transport.clone('knits')
    rel_url = self.texts._index._mapper.map((file_id, None))
    for suffix in ('.kndx', '.knit'):
        try:
            t.delete(rel_url + suffix)
        except _mod_transport.NoSuchFile:
            pass