from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
class TestImportCommit(tests.TestCase):

    def test_commit(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = b'Author'
        mapping = BzrGitMappingv1()
        rev, roundtrip_revid, verifiers = mapping.import_commit(c, mapping.revision_id_foreign_to_bzr)
        self.assertEqual(None, roundtrip_revid)
        self.assertEqual({}, verifiers)
        self.assertEqual('Some message', rev.message)
        self.assertEqual('Committer', rev.committer)
        self.assertEqual('Author', rev.properties['author'])
        self.assertEqual(300, rev.timezone)
        self.assertEqual([], rev.parent_ids)
        self.assertEqual('5', rev.properties['author-timestamp'])
        self.assertEqual('180', rev.properties['author-timezone'])
        self.assertEqual(b'git-v1:' + c.id, rev.revision_id)

    def test_missing_message(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = None
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = b'Author'
        try:
            c.id
        except TypeError:
            self.skipTest('dulwich too old')
        mapping = BzrGitMappingv1()
        rev, roundtrip_revid, verifiers = mapping.import_commit(c, mapping.revision_id_foreign_to_bzr)
        self.assertIs(rev.message, None)

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

    def test_explicit_encoding(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = 'Authér'.encode('iso8859-1')
        c.encoding = b'iso8859-1'
        mapping = BzrGitMappingv1()
        rev, roundtrip_revid, verifiers = mapping.import_commit(c, mapping.revision_id_foreign_to_bzr)
        self.assertEqual(None, roundtrip_revid)
        self.assertEqual({}, verifiers)
        self.assertEqual('Authér', rev.properties['author'])
        self.assertEqual('iso8859-1', rev.properties['git-explicit-encoding'])
        self.assertTrue('git-implicit-encoding' not in rev.properties)

    def test_explicit_encoding_false(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = 'Authér'.encode()
        c.encoding = b'false'
        mapping = BzrGitMappingv1()
        rev, roundtrip_revid, verifiers = mapping.import_commit(c, mapping.revision_id_foreign_to_bzr)
        self.assertEqual(None, roundtrip_revid)
        self.assertEqual({}, verifiers)
        self.assertEqual('Authér', rev.properties['author'])
        self.assertEqual('false', rev.properties['git-explicit-encoding'])
        self.assertTrue('git-implicit-encoding' not in rev.properties)

    def test_implicit_encoding_fallback(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = 'Authér'.encode('latin1')
        mapping = BzrGitMappingv1()
        rev, roundtrip_revid, verifiers = mapping.import_commit(c, mapping.revision_id_foreign_to_bzr)
        self.assertEqual(None, roundtrip_revid)
        self.assertEqual({}, verifiers)
        self.assertEqual('Authér', rev.properties['author'])
        self.assertEqual('latin1', rev.properties['git-implicit-encoding'])
        self.assertTrue('git-explicit-encoding' not in rev.properties)

    def test_implicit_encoding_utf8(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = 'Authér'.encode()
        mapping = BzrGitMappingv1()
        rev, roundtrip_revid, verifiers = mapping.import_commit(c, mapping.revision_id_foreign_to_bzr)
        self.assertEqual(None, roundtrip_revid)
        self.assertEqual({}, verifiers)
        self.assertEqual('Authér', rev.properties['author'])
        self.assertTrue('git-explicit-encoding' not in rev.properties)
        self.assertTrue('git-implicit-encoding' not in rev.properties)

    def test_unknown_extra(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = b'Author'
        c._extra.append((b'iamextra', b'foo'))
        mapping = BzrGitMappingv1()
        self.assertRaises(UnknownCommitExtra, mapping.import_commit, c, mapping.revision_id_foreign_to_bzr)
        mapping.import_commit(c, mapping.revision_id_foreign_to_bzr, strict=False)

    def test_mergetag(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = b'Author'
        tag = make_object(Tag, tagger=b'Jelmer Vernooij <jelmer@samba.org>', name=b'0.1', message=None, object=(Blob, b'd80c186a03f423a81b39df39dc87fd269736ca86'), tag_time=423423423, tag_timezone=0)
        c.mergetag = [tag]
        mapping = BzrGitMappingv1()
        rev, roundtrip_revid, verifiers = mapping.import_commit(c, mapping.revision_id_foreign_to_bzr)
        self.assertEqual(rev.properties['git-mergetag-0'].encode('utf-8'), tag.as_raw_string())

    def test_unknown_hg_fields(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = b'Author'
        c._extra = [(b'HG:extra', b'bla:Foo')]
        mapping = BzrGitMappingv1()
        self.assertRaises(UnknownMercurialCommitExtra, mapping.import_commit, c, mapping.revision_id_foreign_to_bzr)
        mapping.import_commit(c, mapping.revision_id_foreign_to_bzr, strict=False)
        self.assertEqual(mapping.revision_id_foreign_to_bzr(c.id), mapping.get_revision_id(c))

    def test_invalid_utf8(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message \xc1'
        c.committer = b'Committer'
        c.commit_time = 4
        c.author_time = 5
        c.commit_timezone = 60 * 5
        c.author_timezone = 60 * 3
        c.author = b'Author'
        mapping = BzrGitMappingv1()
        self.assertEqual(mapping.revision_id_foreign_to_bzr(c.id), mapping.get_revision_id(c))