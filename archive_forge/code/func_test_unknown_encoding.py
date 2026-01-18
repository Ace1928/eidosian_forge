from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_unknown_encoding(self):
    c = Commit()
    c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
    c.message = b'Some message'
    c.committer = b'Committer'
    c.commit_time = 4
    c.author_time = 5
    c.commit_timezone = 60 * 5
    c.author_timezone = 60 * 3
    c.author = 'Authér'.encode('iso8859-1')
    c.encoding = b'Unknown'
    mapping = BzrGitMappingv1()
    e = self.assertRaises(UnknownCommitEncoding, mapping.import_commit, c, mapping.revision_id_foreign_to_bzr)
    self.assertEqual(e.encoding, 'Unknown')