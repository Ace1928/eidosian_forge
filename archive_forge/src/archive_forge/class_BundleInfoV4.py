import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
class BundleInfoV4:
    """Provide (most of) the BundleInfo interface"""

    def __init__(self, fileobj, serializer):
        self._fileobj = fileobj
        self._serializer = serializer
        self.__real_revisions = None
        self.__revisions = None

    def install(self, repository):
        return self.install_revisions(repository)

    def install_revisions(self, repository, stream_input=True):
        """Install this bundle's revisions into the specified repository

        :param target_repo: The repository to install into
        :param stream_input: If True, will stream input rather than reading it
            all into memory at once.  Reading it into memory all at once is
            (currently) faster.
        """
        with repository.lock_write():
            ri = RevisionInstaller(self.get_bundle_reader(stream_input), self._serializer, repository)
            return ri.install()

    def get_merge_request(self, target_repo):
        """Provide data for performing a merge

        Returns suggested base, suggested target, and patch verification status
        """
        return (None, self.target, 'inapplicable')

    def get_bundle_reader(self, stream_input=True):
        """Return a new BundleReader for the associated bundle

        :param stream_input: If True, the BundleReader stream input rather than
            reading it all into memory at once.  Reading it into memory all at
            once is (currently) faster.
        """
        self._fileobj.seek(0)
        return BundleReader(self._fileobj, stream_input)

    def _get_real_revisions(self):
        if self.__real_revisions is None:
            self.__real_revisions = []
            bundle_reader = self.get_bundle_reader()
            for bytes, metadata, repo_kind, revision_id, file_id in bundle_reader.iter_records():
                if repo_kind == 'info':
                    serializer = self._serializer.get_source_serializer(metadata)
                if repo_kind == 'revision':
                    rev = serializer.read_revision_from_string(bytes)
                    self.__real_revisions.append(rev)
        return self.__real_revisions
    real_revisions = property(_get_real_revisions)

    def _get_revisions(self):
        if self.__revisions is None:
            self.__revisions = []
            for revision in self.real_revisions:
                self.__revisions.append(bundle_data.RevisionInfo.from_revision(revision))
        return self.__revisions
    revisions = property(_get_revisions)

    def _get_target(self):
        return self.revisions[-1].revision_id
    target = property(_get_target)