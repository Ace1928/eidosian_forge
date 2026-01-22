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
class RepositoryFormatPackDevelopment2Subtree(RepositoryFormatPack):
    """A subtrees development repository.

    This format should be retained in 2.3, to provide an upgrade path from this
    to RepositoryFormat2aSubtree.  It can be removed in later releases.

    1.6.1-subtree[as it might have been] with B+Tree indices.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackCommitBuilder
    rich_root_data = True
    experimental = True
    supports_tree_reference = True
    supports_external_lookups = True
    index_builder_class = btree_index.BTreeBuilder
    index_class = btree_index.BTreeGraphIndex

    @property
    def _serializer(self):
        return xml7.serializer_v7

    def _get_matching_bzrdir(self):
        return controldir.format_registry.make_controldir('development5-subtree')

    def _ignore_setting_bzrdir(self, format):
        pass
    _matchingcontroldir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        """See RepositoryFormat.get_format_string()."""
        return b'Bazaar development format 2 with subtree support (needs bzr.dev from before 1.8)\n'

    def get_format_description(self):
        """See RepositoryFormat.get_format_description()."""
        return 'Development repository format, currently the same as 1.6.1-subtree with B+Tree indices.\n'