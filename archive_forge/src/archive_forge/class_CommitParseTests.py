import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
class CommitParseTests(ShaFileCheckTests):

    def make_commit_lines(self, tree=b'd80c186a03f423a81b39df39dc87fd269736ca86', parents=[b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6'], author=default_committer, committer=default_committer, encoding=None, message=b'Merge ../b\n', extra=None):
        lines = []
        if tree is not None:
            lines.append(b'tree ' + tree)
        if parents is not None:
            lines.extend((b'parent ' + p for p in parents))
        if author is not None:
            lines.append(b'author ' + author)
        if committer is not None:
            lines.append(b'committer ' + committer)
        if encoding is not None:
            lines.append(b'encoding ' + encoding)
        if extra is not None:
            for name, value in sorted(extra.items()):
                lines.append(name + b' ' + value)
        lines.append(b'')
        if message is not None:
            lines.append(message)
        return lines

    def make_commit_text(self, **kwargs):
        return b'\n'.join(self.make_commit_lines(**kwargs))

    def test_simple(self):
        c = Commit.from_string(self.make_commit_text())
        self.assertEqual(b'Merge ../b\n', c.message)
        self.assertEqual(b'James Westby <jw+debian@jameswestby.net>', c.author)
        self.assertEqual(b'James Westby <jw+debian@jameswestby.net>', c.committer)
        self.assertEqual(b'd80c186a03f423a81b39df39dc87fd269736ca86', c.tree)
        self.assertEqual([b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6'], c.parents)
        expected_time = datetime.datetime(2007, 3, 24, 22, 1, 59)
        self.assertEqual(expected_time, datetime.datetime.utcfromtimestamp(c.commit_time))
        self.assertEqual(0, c.commit_timezone)
        self.assertEqual(expected_time, datetime.datetime.utcfromtimestamp(c.author_time))
        self.assertEqual(0, c.author_timezone)
        self.assertEqual(None, c.encoding)

    def test_custom(self):
        c = Commit.from_string(self.make_commit_text(extra={b'extra-field': b'data'}))
        self.assertEqual([(b'extra-field', b'data')], c._extra)

    def test_encoding(self):
        c = Commit.from_string(self.make_commit_text(encoding=b'UTF-8'))
        self.assertEqual(b'UTF-8', c.encoding)

    def test_check(self):
        self.assertCheckSucceeds(Commit, self.make_commit_text())
        self.assertCheckSucceeds(Commit, self.make_commit_text(parents=None))
        self.assertCheckSucceeds(Commit, self.make_commit_text(encoding=b'UTF-8'))
        self.assertCheckFails(Commit, self.make_commit_text(tree=b'xxx'))
        self.assertCheckFails(Commit, self.make_commit_text(parents=[a_sha, b'xxx']))
        bad_committer = b'some guy without an email address 1174773719 +0000'
        self.assertCheckFails(Commit, self.make_commit_text(committer=bad_committer))
        self.assertCheckFails(Commit, self.make_commit_text(author=bad_committer))
        self.assertCheckFails(Commit, self.make_commit_text(author=None))
        self.assertCheckFails(Commit, self.make_commit_text(committer=None))
        self.assertCheckFails(Commit, self.make_commit_text(author=None, committer=None))

    def test_check_duplicates(self):
        for i in range(5):
            lines = self.make_commit_lines(parents=[a_sha], encoding=b'UTF-8')
            lines.insert(i, lines[i])
            text = b'\n'.join(lines)
            if lines[i].startswith(b'parent'):
                self.assertCheckSucceeds(Commit, text)
            else:
                self.assertCheckFails(Commit, text)

    def test_check_order(self):
        lines = self.make_commit_lines(parents=[a_sha], encoding=b'UTF-8')
        headers = lines[:5]
        rest = lines[5:]
        for perm in permutations(headers):
            perm = list(perm)
            text = b'\n'.join(perm + rest)
            if perm == headers:
                self.assertCheckSucceeds(Commit, text)
            else:
                self.assertCheckFails(Commit, text)

    def test_check_commit_with_unparseable_time(self):
        identity_with_wrong_time = b'Igor Sysoev <igor@sysoev.ru> 18446743887488505614+42707004'
        self.assertCheckFails(Commit, self.make_commit_text(author=default_committer, committer=identity_with_wrong_time))
        self.assertCheckFails(Commit, self.make_commit_text(author=identity_with_wrong_time, committer=default_committer))

    def test_check_commit_with_overflow_date(self):
        """Date with overflow should raise an ObjectFormatException when checked."""
        identity_with_wrong_time = b'Igor Sysoev <igor@sysoev.ru> 18446743887488505614 +42707004'
        commit0 = Commit.from_string(self.make_commit_text(author=identity_with_wrong_time, committer=default_committer))
        commit1 = Commit.from_string(self.make_commit_text(author=default_committer, committer=identity_with_wrong_time))
        for commit in [commit0, commit1]:
            with self.assertRaises(ObjectFormatException):
                commit.check()

    def test_mangled_author_line(self):
        """Mangled author line should successfully parse."""
        author_line = b'Karl MacMillan <kmacmill@redhat.com> <"Karl MacMillan <kmacmill@redhat.com>"> 1197475547 -0500'
        expected_identity = b'Karl MacMillan <kmacmill@redhat.com> <"Karl MacMillan <kmacmill@redhat.com>">'
        commit = Commit.from_string(self.make_commit_text(author=author_line))
        self.assertEqual(commit.author, expected_identity)
        with self.assertRaises(ObjectFormatException):
            commit.check()

    def test_parse_gpgsig(self):
        c = Commit.from_string(b'tree aaff74984cccd156a469afa7d9ab10e4777beb24\nauthor Jelmer Vernooij <jelmer@samba.org> 1412179807 +0200\ncommitter Jelmer Vernooij <jelmer@samba.org> 1412179807 +0200\ngpgsig -----BEGIN PGP SIGNATURE-----\n Version: GnuPG v1\n \n iQIcBAABCgAGBQJULCdfAAoJEACAbyvXKaRXuKwP/RyP9PA49uAvu8tQVCC/uBa8\n vi975+xvO14R8Pp8k2nps7lSxCdtCd+xVT1VRHs0wNhOZo2YCVoU1HATkPejqSeV\n NScTHcxnk4/+bxyfk14xvJkNp7FlQ3npmBkA+lbV0Ubr33rvtIE5jiJPyz+SgWAg\n xdBG2TojV0squj00GoH/euK6aX7GgZtwdtpTv44haCQdSuPGDcI4TORqR6YSqvy3\n GPE+3ZqXPFFb+KILtimkxitdwB7CpwmNse2vE3rONSwTvi8nq3ZoQYNY73CQGkUy\n qoFU0pDtw87U3niFin1ZccDgH0bB6624sLViqrjcbYJeg815Htsu4rmzVaZADEVC\n XhIO4MThebusdk0AcNGjgpf3HRHk0DPMDDlIjm+Oao0cqovvF6VyYmcb0C+RmhJj\n dodLXMNmbqErwTk3zEkW0yZvNIYXH7m9SokPCZa4eeIM7be62X6h1mbt0/IU6Th+\n v18fS0iTMP/Viug5und+05C/v04kgDo0CPphAbXwWMnkE4B6Tl9sdyUYXtvQsL7x\n 0+WP1gL27ANqNZiI07Kz/BhbBAQI/+2TFT7oGr0AnFPQ5jHp+3GpUf6OKuT1wT3H\n ND189UFuRuubxb42vZhpcXRbqJVWnbECTKVUPsGZqat3enQUB63uM4i6/RdONDZA\n fDeF1m4qYs+cUXKNUZ03\n =X6RT\n -----END PGP SIGNATURE-----\n\nfoo\n')
        self.assertEqual(b'foo\n', c.message)
        self.assertEqual([], c._extra)
        self.assertEqual(b'-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1\n\niQIcBAABCgAGBQJULCdfAAoJEACAbyvXKaRXuKwP/RyP9PA49uAvu8tQVCC/uBa8\nvi975+xvO14R8Pp8k2nps7lSxCdtCd+xVT1VRHs0wNhOZo2YCVoU1HATkPejqSeV\nNScTHcxnk4/+bxyfk14xvJkNp7FlQ3npmBkA+lbV0Ubr33rvtIE5jiJPyz+SgWAg\nxdBG2TojV0squj00GoH/euK6aX7GgZtwdtpTv44haCQdSuPGDcI4TORqR6YSqvy3\nGPE+3ZqXPFFb+KILtimkxitdwB7CpwmNse2vE3rONSwTvi8nq3ZoQYNY73CQGkUy\nqoFU0pDtw87U3niFin1ZccDgH0bB6624sLViqrjcbYJeg815Htsu4rmzVaZADEVC\nXhIO4MThebusdk0AcNGjgpf3HRHk0DPMDDlIjm+Oao0cqovvF6VyYmcb0C+RmhJj\ndodLXMNmbqErwTk3zEkW0yZvNIYXH7m9SokPCZa4eeIM7be62X6h1mbt0/IU6Th+\nv18fS0iTMP/Viug5und+05C/v04kgDo0CPphAbXwWMnkE4B6Tl9sdyUYXtvQsL7x\n0+WP1gL27ANqNZiI07Kz/BhbBAQI/+2TFT7oGr0AnFPQ5jHp+3GpUf6OKuT1wT3H\nND189UFuRuubxb42vZhpcXRbqJVWnbECTKVUPsGZqat3enQUB63uM4i6/RdONDZA\nfDeF1m4qYs+cUXKNUZ03\n=X6RT\n-----END PGP SIGNATURE-----', c.gpgsig)

    def test_parse_header_trailing_newline(self):
        c = Commit.from_string(b'tree a7d6277f78d3ecd0230a1a5df6db00b1d9c521ac\nparent c09b6dec7a73760fbdb478383a3c926b18db8bbe\nauthor Neil Matatall <oreoshake@github.com> 1461964057 -1000\ncommitter Neil Matatall <oreoshake@github.com> 1461964057 -1000\ngpgsig -----BEGIN PGP SIGNATURE-----\n \n wsBcBAABCAAQBQJXI80ZCRA6pcNDcVZ70gAAarcIABs72xRX3FWeox349nh6ucJK\n CtwmBTusez2Zwmq895fQEbZK7jpaGO5TRO4OvjFxlRo0E08UFx3pxZHSpj6bsFeL\n hHsDXnCaotphLkbgKKRdGZo7tDqM84wuEDlh4MwNe7qlFC7bYLDyysc81ZX5lpMm\n 2MFF1TvjLAzSvkT7H1LPkuR3hSvfCYhikbPOUNnKOo0sYjeJeAJ/JdAVQ4mdJIM0\n gl3REp9+A+qBEpNQI7z94Pg5Bc5xenwuDh3SJgHvJV6zBWupWcdB3fAkVd4TPnEZ\n nHxksHfeNln9RKseIDcy4b2ATjhDNIJZARHNfr6oy4u3XPW4svRqtBsLoMiIeuI=\n =ms6q\n -----END PGP SIGNATURE-----\n \n\n3.3.0 version bump and docs\n')
        self.assertEqual([], c._extra)
        self.assertEqual(b'-----BEGIN PGP SIGNATURE-----\n\nwsBcBAABCAAQBQJXI80ZCRA6pcNDcVZ70gAAarcIABs72xRX3FWeox349nh6ucJK\nCtwmBTusez2Zwmq895fQEbZK7jpaGO5TRO4OvjFxlRo0E08UFx3pxZHSpj6bsFeL\nhHsDXnCaotphLkbgKKRdGZo7tDqM84wuEDlh4MwNe7qlFC7bYLDyysc81ZX5lpMm\n2MFF1TvjLAzSvkT7H1LPkuR3hSvfCYhikbPOUNnKOo0sYjeJeAJ/JdAVQ4mdJIM0\ngl3REp9+A+qBEpNQI7z94Pg5Bc5xenwuDh3SJgHvJV6zBWupWcdB3fAkVd4TPnEZ\nnHxksHfeNln9RKseIDcy4b2ATjhDNIJZARHNfr6oy4u3XPW4svRqtBsLoMiIeuI=\n=ms6q\n-----END PGP SIGNATURE-----\n', c.gpgsig)