import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
def _copy_signature_texts(self):
    source_vf, target_vf = self._build_vfs('signature', False, False)
    signature_keys = source_vf.keys()
    signature_keys.intersection(self.revision_keys)
    self._copy_stream(source_vf, target_vf, signature_keys, 'signatures', self._get_progress_stream, 5)