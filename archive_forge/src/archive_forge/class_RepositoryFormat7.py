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
class RepositoryFormat7(MetaDirVersionedFileRepositoryFormat):
    """Bzr repository 7.

    This repository format has:
     - weaves for file texts and inventory
     - hash subdirectory based stores.
     - TextStores for revisions and signatures.
     - a format marker of its own
     - an optional 'shared-storage' flag
     - an optional 'no-working-trees' flag
    """
    _versionedfile_class = weave.WeaveFile
    supports_ghosts = False
    supports_chks = False
    supports_funky_characters = False
    revision_graph_can_have_wrong_parents = False
    _fetch_order = 'topological'
    _fetch_reconcile = True
    fast_deltas = False

    @property
    def _serializer(self):
        return xml5.serializer_v5

    @classmethod
    def get_format_string(cls):
        """See RepositoryFormat.get_format_string()."""
        return b'Bazaar-NG Repository format 7'

    def get_format_description(self):
        """See RepositoryFormat.get_format_description()."""
        return 'Weave repository format 7'

    def _get_inventories(self, repo_transport, repo, name='inventory'):
        mapper = versionedfile.ConstantMapper(name)
        return versionedfile.ThunkedVersionedFiles(repo_transport, weave.WeaveFile, mapper, repo.is_locked)

    def _get_revisions(self, repo_transport, repo):
        return RevisionTextStore(repo_transport.clone('revision-store'), xml5.serializer_v5, True, versionedfile.HashPrefixMapper(), repo.is_locked, repo.is_write_locked)

    def _get_signatures(self, repo_transport, repo):
        return SignatureTextStore(repo_transport.clone('revision-store'), True, versionedfile.HashPrefixMapper(), repo.is_locked, repo.is_write_locked)

    def _get_texts(self, repo_transport, repo):
        mapper = versionedfile.HashPrefixMapper()
        base_transport = repo_transport.clone('weaves')
        return versionedfile.ThunkedVersionedFiles(base_transport, weave.WeaveFile, mapper, repo.is_locked)

    def initialize(self, a_controldir, shared=False):
        """Create a weave repository.

        :param shared: If true the repository will be initialized as a shared
                       repository.
        """
        sio = BytesIO()
        weavefile.write_weave_v5(weave.Weave(), sio)
        empty_weave = sio.getvalue()
        trace.mutter('creating repository in %s.', a_controldir.transport.base)
        dirs = ['revision-store', 'weaves']
        files = [('inventory.weave', BytesIO(empty_weave))]
        utf8_files = [('format', self.get_format_string())]
        self._upload_blank_content(a_controldir, dirs, files, utf8_files, shared)
        return self.open(a_controldir=a_controldir, _found=True)

    def open(self, a_controldir, _found=False, _override_transport=None):
        """See RepositoryFormat.open().

        :param _override_transport: INTERNAL USE ONLY. Allows opening the
                                    repository at a slightly different url
                                    than normal. I.e. during 'upgrade'.
        """
        if not _found:
            format = RepositoryFormatMetaDir.find_format(a_controldir)
        if _override_transport is not None:
            repo_transport = _override_transport
        else:
            repo_transport = a_controldir.get_repository_transport(None)
        control_files = lockable_files.LockableFiles(repo_transport, 'lock', lockdir.LockDir)
        result = WeaveMetaDirRepository(_format=self, a_controldir=a_controldir, control_files=control_files)
        result.revisions = self._get_revisions(repo_transport, result)
        result.signatures = self._get_signatures(repo_transport, result)
        result.inventories = self._get_inventories(repo_transport, result)
        result.texts = self._get_texts(repo_transport, result)
        result.chk_bytes = None
        result._transport = repo_transport
        return result

    def is_deprecated(self):
        return True