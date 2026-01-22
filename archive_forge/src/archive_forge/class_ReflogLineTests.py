from io import BytesIO
from dulwich.tests import TestCase
from ..objects import ZERO_SHA
from ..reflog import (
class ReflogLineTests(TestCase):

    def test_format(self):
        self.assertEqual(b'0000000000000000000000000000000000000000 49030649db3dfec5a9bc03e5dde4255a14499f16 Jelmer Vernooij <jelmer@jelmer.uk> 1446552482 +0000\tclone: from git://jelmer.uk/samba', format_reflog_line(b'0000000000000000000000000000000000000000', b'49030649db3dfec5a9bc03e5dde4255a14499f16', b'Jelmer Vernooij <jelmer@jelmer.uk>', 1446552482, 0, b'clone: from git://jelmer.uk/samba'))
        self.assertEqual(b'0000000000000000000000000000000000000000 49030649db3dfec5a9bc03e5dde4255a14499f16 Jelmer Vernooij <jelmer@jelmer.uk> 1446552482 +0000\tclone: from git://jelmer.uk/samba', format_reflog_line(None, b'49030649db3dfec5a9bc03e5dde4255a14499f16', b'Jelmer Vernooij <jelmer@jelmer.uk>', 1446552482, 0, b'clone: from git://jelmer.uk/samba'))

    def test_parse(self):
        reflog_line = b'0000000000000000000000000000000000000000 49030649db3dfec5a9bc03e5dde4255a14499f16 Jelmer Vernooij <jelmer@jelmer.uk> 1446552482 +0000\tclone: from git://jelmer.uk/samba'
        self.assertEqual((b'0000000000000000000000000000000000000000', b'49030649db3dfec5a9bc03e5dde4255a14499f16', b'Jelmer Vernooij <jelmer@jelmer.uk>', 1446552482, 0, b'clone: from git://jelmer.uk/samba'), parse_reflog_line(reflog_line))