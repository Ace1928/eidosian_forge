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
class RepositoryFormat2aSubtree(RepositoryFormat2a):
    """A 2a repository format that supports nested trees.

    """

    def _get_matching_bzrdir(self):
        return controldir.format_registry.make_controldir('development-subtree')

    def _ignore_setting_bzrdir(self, format):
        pass
    _matchingcontroldir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        return b'Bazaar development format 8\n'

    def get_format_description(self):
        """See RepositoryFormat.get_format_description()."""
        return 'Development repository format 8 - nested trees, group compression and chk inventories'
    experimental = True
    supports_tree_reference = True