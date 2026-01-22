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
class RepositoryFormatKnit3(RepositoryFormatKnit):
    """Bzr repository knit format 3.

    This repository format has:
     - knits for file texts and inventory
     - hash subdirectory based stores.
     - knits for revisions and signatures
     - TextStores for revisions and signatures.
     - a format marker of its own
     - an optional 'shared-storage' flag
     - an optional 'no-working-trees' flag
     - a LockDir lock
     - support for recording full info about the tree root
     - support for recording tree-references
    """
    repository_class = KnitRepository
    _commit_builder_class = VersionedFileCommitBuilder
    rich_root_data = True
    experimental = True
    supports_tree_reference = True

    @property
    def _serializer(self):
        return xml7.serializer_v7

    def _get_matching_bzrdir(self):
        return controldir.format_registry.make_controldir('dirstate-with-subtree')

    def _ignore_setting_bzrdir(self, format):
        pass
    _matchingcontroldir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        """See RepositoryFormat.get_format_string()."""
        return b'Bazaar Knit Repository Format 3 (bzr 0.15)\n'

    def get_format_description(self):
        """See RepositoryFormat.get_format_description()."""
        return 'Knit repository format 3'