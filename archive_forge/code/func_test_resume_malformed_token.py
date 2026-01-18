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
def test_resume_malformed_token(self):
    self.vfs_transport_factory = memory.MemoryServer
    repo = self.make_repository('repo', format=self.get_format())
    token = self._lock_write(repo).repository_token
    repo.start_write_group()
    text_key = (b'file-id', b'revid')
    repo.texts.add_lines(text_key, (), [b'lines'])
    wg_tokens = repo.suspend_write_group()
    new_repo = self.make_repository('new_repo', format=self.get_format())
    token = self._lock_write(new_repo).repository_token
    hacked_wg_token = '../../../../repo/.bzr/repository/upload/' + wg_tokens[0]
    self.assertRaises(errors.UnresumableWriteGroup, new_repo.resume_write_group, [hacked_wg_token])