import io
import os
import shutil
import sys
import tempfile
from dulwich.tests import SkipTest, TestCase
from ..file import FileLocked, GitFile, _fancy_rename
class GitFileTests(TestCase):

    def setUp(self):
        super().setUp()
        self._tempdir = tempfile.mkdtemp()
        f = open(self.path('foo'), 'wb')
        f.write(b'foo contents')
        f.close()

    def tearDown(self):
        shutil.rmtree(self._tempdir)
        super().tearDown()

    def path(self, filename):
        return os.path.join(self._tempdir, filename)

    def test_invalid(self):
        foo = self.path('foo')
        self.assertRaises(IOError, GitFile, foo, mode='r')
        self.assertRaises(IOError, GitFile, foo, mode='ab')
        self.assertRaises(IOError, GitFile, foo, mode='r+b')
        self.assertRaises(IOError, GitFile, foo, mode='w+b')
        self.assertRaises(IOError, GitFile, foo, mode='a+bU')

    def test_readonly(self):
        f = GitFile(self.path('foo'), 'rb')
        self.assertIsInstance(f, io.IOBase)
        self.assertEqual(b'foo contents', f.read())
        self.assertEqual(b'', f.read())
        f.seek(4)
        self.assertEqual(b'contents', f.read())
        f.close()

    def test_default_mode(self):
        f = GitFile(self.path('foo'))
        self.assertEqual(b'foo contents', f.read())
        f.close()

    def test_write(self):
        foo = self.path('foo')
        foo_lock = '%s.lock' % foo
        orig_f = open(foo, 'rb')
        self.assertEqual(orig_f.read(), b'foo contents')
        orig_f.close()
        self.assertFalse(os.path.exists(foo_lock))
        f = GitFile(foo, 'wb')
        self.assertFalse(f.closed)
        self.assertRaises(AttributeError, getattr, f, 'not_a_file_property')
        self.assertTrue(os.path.exists(foo_lock))
        f.write(b'new stuff')
        f.seek(4)
        f.write(b'contents')
        f.close()
        self.assertFalse(os.path.exists(foo_lock))
        new_f = open(foo, 'rb')
        self.assertEqual(b'new contents', new_f.read())
        new_f.close()

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

    def test_abort(self):
        foo = self.path('foo')
        foo_lock = '%s.lock' % foo
        orig_f = open(foo, 'rb')
        self.assertEqual(orig_f.read(), b'foo contents')
        orig_f.close()
        f = GitFile(foo, 'wb')
        f.write(b'new contents')
        f.abort()
        self.assertTrue(f.closed)
        self.assertFalse(os.path.exists(foo_lock))
        new_orig_f = open(foo, 'rb')
        self.assertEqual(new_orig_f.read(), b'foo contents')
        new_orig_f.close()

    def test_abort_close(self):
        foo = self.path('foo')
        f = GitFile(foo, 'wb')
        f.abort()
        try:
            f.close()
        except OSError:
            self.fail()
        f = GitFile(foo, 'wb')
        f.close()
        try:
            f.abort()
        except OSError:
            self.fail()

    def test_abort_close_removed(self):
        foo = self.path('foo')
        f = GitFile(foo, 'wb')
        f._file.close()
        os.remove(foo + '.lock')
        f.abort()
        self.assertTrue(f._closed)