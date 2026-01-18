import io
import os
import shutil
import sys
import tempfile
from dulwich.tests import SkipTest, TestCase
from ..file import FileLocked, GitFile, _fancy_rename
def test_open_twice(self):
    foo = self.path('foo')
    f1 = GitFile(foo, 'wb')
    f1.write(b'new')
    try:
        f2 = GitFile(foo, 'wb')
        self.fail()
    except FileLocked:
        pass
    else:
        f2.close()
    f1.write(b' contents')
    f1.close()
    f = open(foo, 'rb')
    self.assertEqual(b'new contents', f.read())
    f.close()