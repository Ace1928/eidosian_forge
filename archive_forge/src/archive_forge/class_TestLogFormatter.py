import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestLogFormatter(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.rev = revision.Revision(b'a-id')
        self.lf = log.LogFormatter(None)

    def test_short_committer(self):

        def assertCommitter(expected, committer):
            self.rev.committer = committer
            self.assertEqual(expected, self.lf.short_committer(self.rev))
        assertCommitter('John Doe', 'John Doe <jdoe@example.com>')
        assertCommitter('John Smith', 'John Smith <jsmith@example.com>')
        assertCommitter('John Smith', 'John Smith')
        assertCommitter('jsmith@example.com', 'jsmith@example.com')
        assertCommitter('jsmith@example.com', '<jsmith@example.com>')
        assertCommitter('John Smith', 'John Smith jsmith@example.com')

    def test_short_author(self):

        def assertAuthor(expected, author):
            self.rev.properties['author'] = author
            self.assertEqual(expected, self.lf.short_author(self.rev))
        assertAuthor('John Smith', 'John Smith <jsmith@example.com>')
        assertAuthor('John Smith', 'John Smith')
        assertAuthor('jsmith@example.com', 'jsmith@example.com')
        assertAuthor('jsmith@example.com', '<jsmith@example.com>')
        assertAuthor('John Smith', 'John Smith jsmith@example.com')

    def test_short_author_from_committer(self):
        self.rev.committer = 'John Doe <jdoe@example.com>'
        self.assertEqual('John Doe', self.lf.short_author(self.rev))

    def test_short_author_from_authors(self):
        self.rev.properties['authors'] = 'John Smith <jsmith@example.com>\nJane Rey <jrey@example.com>'
        self.assertEqual('John Smith', self.lf.short_author(self.rev))