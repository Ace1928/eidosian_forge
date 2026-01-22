import sys
from breezy import errors, osutils, repository
from breezy.bzr import inventory, versionedfile
from breezy.bzr.vf_search import SearchResult
from breezy.errors import NoSuchRevision
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable
from breezy.tests.per_interrepository import TestCaseWithInterRepository
from breezy.tests.per_interrepository.test_interrepository import \
def assertCanStreamRevision(self, repo, revision_id):
    exclude_keys = set(repo.all_revision_ids()) - {revision_id}
    search = SearchResult([revision_id], exclude_keys, 1, [revision_id])
    source = repo._get_source(repo._format)
    for substream_kind, substream in source.get_stream(search):
        list(substream)