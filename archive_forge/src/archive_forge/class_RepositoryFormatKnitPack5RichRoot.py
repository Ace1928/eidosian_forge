from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
class RepositoryFormatKnitPack5RichRoot(RepositoryFormatPack):
    """A repository with rich roots and stacking.

    Supports stacking on other repositories, allowing data to be accessed
    without being stored locally.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackCommitBuilder
    rich_root_data = True
    supports_tree_reference = False
    supports_external_lookups = True
    index_builder_class = InMemoryGraphIndex
    index_class = GraphIndex

    @property
    def _serializer(self):
        return xml6.serializer_v6

    def _get_matching_bzrdir(self):
        return controldir.format_registry.make_controldir('1.6.1-rich-root')

    def _ignore_setting_bzrdir(self, format):
        pass
    _matchingcontroldir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        """See RepositoryFormat.get_format_string()."""
        return b'Bazaar RepositoryFormatKnitPack5RichRoot (bzr 1.6.1)\n'

    def get_format_description(self):
        return 'Packs 5 rich-root (adds stacking support, requires bzr 1.6.1)'