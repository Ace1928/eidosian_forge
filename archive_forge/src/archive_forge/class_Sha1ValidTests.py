from ...tests import TestCase
from ..revspec import valid_git_sha1
class Sha1ValidTests(TestCase):

    def test_invalid(self):
        self.assertFalse(valid_git_sha1(b'git-v1:abcde'))

    def test_valid(self):
        self.assertTrue(valid_git_sha1(b'aabbccddee'))
        self.assertTrue(valid_git_sha1(b'aabbccd'))