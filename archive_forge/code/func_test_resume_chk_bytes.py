from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def test_resume_chk_bytes(self):
    self.vfs_transport_factory = memory.MemoryServer
    repo = self.make_repository('repo', format=self.get_format())
    if repo.chk_bytes is None:
        raise TestNotApplicable('no chk_bytes for this repository')
    token = self._lock_write(repo).repository_token
    repo.start_write_group()
    text = b'a bit of text\n'
    key = (b'sha1:' + osutils.sha_string(text),)
    repo.chk_bytes.add_lines(key, (), [text])
    wg_tokens = repo.suspend_write_group()
    same_repo = repo.controldir.open_repository()
    same_repo.lock_write()
    self.addCleanup(same_repo.unlock)
    same_repo.resume_write_group(wg_tokens)
    self.assertEqual([key], list(same_repo.chk_bytes.keys()))
    self.assertEqual(text, next(same_repo.chk_bytes.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext'))
    same_repo.abort_write_group()
    self.assertEqual([], list(same_repo.chk_bytes.keys()))