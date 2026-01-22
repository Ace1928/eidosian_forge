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
class RepositoryFormatKnit(MetaDirVersionedFileRepositoryFormat):
    """Bzr repository knit format (generalized).

    This repository format has:
     - knits for file texts and inventory
     - hash subdirectory based stores.
     - knits for revisions and signatures
     - TextStores for revisions and signatures.
     - a format marker of its own
     - an optional 'shared-storage' flag
     - an optional 'no-working-trees' flag
     - a LockDir lock
    """
    repository_class: Type[Repository]
    _commit_builder_class: Type[VersionedFileCommitBuilder]

    @property
    def _serializer(self):
        return xml5.serializer_v5
    supports_ghosts = True
    supports_external_lookups = False
    supports_chks = False
    _fetch_order = 'topological'
    _fetch_uses_deltas = True
    fast_deltas = False
    supports_funky_characters = True
    revision_graph_can_have_wrong_parents = True

    def _get_inventories(self, repo_transport, repo, name='inventory'):
        mapper = versionedfile.ConstantMapper(name)
        index = _mod_knit._KndxIndex(repo_transport, mapper, repo.get_transaction, repo.is_write_locked, repo.is_locked)
        access = _mod_knit._KnitKeyAccess(repo_transport, mapper)
        return _mod_knit.KnitVersionedFiles(index, access, annotated=False)

    def _get_revisions(self, repo_transport, repo):
        mapper = versionedfile.ConstantMapper('revisions')
        index = _mod_knit._KndxIndex(repo_transport, mapper, repo.get_transaction, repo.is_write_locked, repo.is_locked)
        access = _mod_knit._KnitKeyAccess(repo_transport, mapper)
        return _mod_knit.KnitVersionedFiles(index, access, max_delta_chain=0, annotated=False)

    def _get_signatures(self, repo_transport, repo):
        mapper = versionedfile.ConstantMapper('signatures')
        index = _mod_knit._KndxIndex(repo_transport, mapper, repo.get_transaction, repo.is_write_locked, repo.is_locked)
        access = _mod_knit._KnitKeyAccess(repo_transport, mapper)
        return _mod_knit.KnitVersionedFiles(index, access, max_delta_chain=0, annotated=False)

    def _get_texts(self, repo_transport, repo):
        mapper = versionedfile.HashEscapedPrefixMapper()
        base_transport = repo_transport.clone('knits')
        index = _mod_knit._KndxIndex(base_transport, mapper, repo.get_transaction, repo.is_write_locked, repo.is_locked)
        access = _mod_knit._KnitKeyAccess(base_transport, mapper)
        return _mod_knit.KnitVersionedFiles(index, access, max_delta_chain=200, annotated=True)

    def initialize(self, a_controldir, shared=False):
        """Create a knit format 1 repository.

        :param a_controldir: bzrdir to contain the new repository; must already
            be initialized.
        :param shared: If true the repository will be initialized as a shared
                       repository.
        """
        trace.mutter('creating repository in %s.', a_controldir.transport.base)
        dirs = ['knits']
        files = []
        utf8_files = [('format', self.get_format_string())]
        self._upload_blank_content(a_controldir, dirs, files, utf8_files, shared)
        repo_transport = a_controldir.get_repository_transport(None)
        control_files = lockable_files.LockableFiles(repo_transport, 'lock', lockdir.LockDir)
        transaction = transactions.WriteTransaction()
        result = self.open(a_controldir=a_controldir, _found=True)
        result.lock_write()
        result.inventories.get_parent_map([(b'A',)])
        result.revisions.get_parent_map([(b'A',)])
        result.signatures.get_parent_map([(b'A',)])
        result.unlock()
        self._run_post_repo_init_hooks(result, a_controldir, shared)
        return result

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
        repo = self.repository_class(_format=self, a_controldir=a_controldir, control_files=control_files, _commit_builder_class=self._commit_builder_class, _serializer=self._serializer)
        repo.revisions = self._get_revisions(repo_transport, repo)
        repo.signatures = self._get_signatures(repo_transport, repo)
        repo.inventories = self._get_inventories(repo_transport, repo)
        repo.texts = self._get_texts(repo_transport, repo)
        repo.chk_bytes = None
        repo._transport = repo_transport
        return repo