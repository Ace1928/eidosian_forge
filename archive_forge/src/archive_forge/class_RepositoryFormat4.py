import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
class RepositoryFormat4(PreSplitOutRepositoryFormat):
    """Bzr repository format 4.

    This repository format has:
     - flat stores
     - TextStores for texts, inventories,revisions.

    This format is deprecated: it indexes texts using a text id which is
    removed in format 5; initialization and write support for this format
    has been removed.
    """
    supports_funky_characters = False
    _matchingcontroldir = weave_bzrdir.BzrDirFormat4()

    def get_format_description(self):
        """See RepositoryFormat.get_format_description()."""
        return 'Repository format 4'

    def initialize(self, url, shared=False, _internal=False):
        """Format 4 branches cannot be created."""
        raise errors.UninitializableFormat(self)

    def is_supported(self):
        """Format 4 is not supported.

        It is not supported because the model changed from 4 to 5 and the
        conversion logic is expensive - so doing it on the fly was not
        feasible.
        """
        return False

    def _get_inventories(self, repo_transport, repo, name='inventory'):
        return None

    def _get_revisions(self, repo_transport, repo):
        from .xml4 import serializer_v4
        return RevisionTextStore(repo_transport.clone('revision-store'), serializer_v4, True, versionedfile.PrefixMapper(), repo.is_locked, repo.is_write_locked)

    def _get_signatures(self, repo_transport, repo):
        return SignatureTextStore(repo_transport.clone('revision-store'), False, versionedfile.PrefixMapper(), repo.is_locked, repo.is_write_locked)

    def _get_texts(self, repo_transport, repo):
        return None