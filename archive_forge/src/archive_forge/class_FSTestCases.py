from __future__ import absolute_import, unicode_literals
import io
import itertools
import json
import os
import six
import time
import unittest
import warnings
from datetime import datetime
from six import text_type
import fs.copy
import fs.move
from fs import ResourceType, Seek, errors, glob, walk
from fs.opener import open_fs
from fs.subfs import ClosingSubFS, SubFS
class FSTestCases(object):
    """Basic FS tests."""
    data1 = b'foo' * 256 * 1024
    data2 = b'bar' * 2 * 256 * 1024
    data3 = b'baz' * 3 * 256 * 1024
    data4 = b'egg' * 7 * 256 * 1024

    def make_fs(self):
        """Return an FS instance."""
        raise NotImplementedError('implement me')

    def destroy_fs(self, fs):
        """Destroy a FS instance.

        Arguments:
            fs (FS): A filesystem instance previously opened
                by `~fs.test.FSTestCases.make_fs`.

        """
        fs.close()

    def setUp(self):
        self.fs = self.make_fs()

    def tearDown(self):
        self.destroy_fs(self.fs)
        del self.fs

    def assert_exists(self, path):
        """Assert a path exists.

        Arguments:
            path (str): A path on the filesystem.

        """
        self.assertTrue(self.fs.exists(path))

    def assert_not_exists(self, path):
        """Assert a path does not exist.

        Arguments:
            path (str): A path on the filesystem.

        """
        self.assertFalse(self.fs.exists(path))

    def assert_isempty(self, path):
        """Assert a path is an empty directory.

        Arguments:
            path (str): A path on the filesystem.

        """
        self.assertTrue(self.fs.isempty(path))

    def assert_isfile(self, path):
        """Assert a path is a file.

        Arguments:
            path (str): A path on the filesystem.

        """
        self.assertTrue(self.fs.isfile(path))

    def assert_isdir(self, path):
        """Assert a path is a directory.

        Arguments:
            path (str): A path on the filesystem.

        """
        self.assertTrue(self.fs.isdir(path))

    def assert_bytes(self, path, contents):
        """Assert a file contains the given bytes.

        Arguments:
            path (str): A path on the filesystem.
            contents (bytes): Bytes to compare.

        """
        assert isinstance(contents, bytes)
        data = self.fs.readbytes(path)
        self.assertEqual(data, contents)
        self.assertIsInstance(data, bytes)

    def assert_text(self, path, contents):
        """Assert a file contains the given text.

        Arguments:
            path (str): A path on the filesystem.
            contents (str): Text to compare.

        """
        assert isinstance(contents, text_type)
        with self.fs.open(path, 'rt') as f:
            data = f.read()
        self.assertEqual(data, contents)
        self.assertIsInstance(data, text_type)

    def test_root_dir(self):
        with self.assertRaises(errors.FileExpected):
            self.fs.open('/')
        with self.assertRaises(errors.FileExpected):
            self.fs.openbin('/')

    def test_appendbytes(self):
        with self.assertRaises(TypeError):
            self.fs.appendbytes('foo', 'bar')
        self.fs.appendbytes('foo', b'bar')
        self.assert_bytes('foo', b'bar')
        self.fs.appendbytes('foo', b'baz')
        self.assert_bytes('foo', b'barbaz')

    def test_appendtext(self):
        with self.assertRaises(TypeError):
            self.fs.appendtext('foo', b'bar')
        self.fs.appendtext('foo', 'bar')
        self.assert_text('foo', 'bar')
        self.fs.appendtext('foo', 'baz')
        self.assert_text('foo', 'barbaz')

    def test_basic(self):
        repr(self.fs)
        self.assertIsInstance(six.text_type(self.fs), six.text_type)

    def test_getmeta(self):
        meta = self.fs.getmeta()
        self.assertEqual(meta, self.fs.getmeta(namespace='standard'))
        self.assertTrue(isinstance(meta, dict))
        no_meta = self.fs.getmeta('__nosuchnamespace__')
        self.assertIsInstance(no_meta, dict)
        self.assertFalse(no_meta)

    def test_isfile(self):
        self.assertFalse(self.fs.isfile('foo.txt'))
        self.fs.create('foo.txt')
        self.assertTrue(self.fs.isfile('foo.txt'))
        self.fs.makedir('bar')
        self.assertFalse(self.fs.isfile('bar'))

    def test_isdir(self):
        self.assertFalse(self.fs.isdir('foo'))
        self.fs.create('bar')
        self.fs.makedir('foo')
        self.assertTrue(self.fs.isdir('foo'))
        self.assertFalse(self.fs.isdir('bar'))

    def test_islink(self):
        self.fs.touch('foo')
        self.assertFalse(self.fs.islink('foo'))
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.islink('bar')

    def test_getsize(self):
        self.fs.writebytes('empty', b'')
        self.fs.writebytes('one', b'a')
        self.fs.writebytes('onethousand', ('b' * 1000).encode('ascii'))
        self.assertEqual(self.fs.getsize('empty'), 0)
        self.assertEqual(self.fs.getsize('one'), 1)
        self.assertEqual(self.fs.getsize('onethousand'), 1000)
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.getsize('doesnotexist')

    def test_getsyspath(self):
        self.fs.create('foo')
        try:
            syspath = self.fs.getsyspath('foo')
        except errors.NoSysPath:
            self.assertFalse(self.fs.hassyspath('foo'))
        else:
            self.assertIsInstance(syspath, text_type)
            self.assertIsInstance(self.fs.getospath('foo'), bytes)
            self.assertTrue(self.fs.hassyspath('foo'))
        self.fs.hassyspath('a/b/c/foo/bar')

    def test_geturl(self):
        self.fs.create('foo')
        try:
            self.fs.geturl('foo')
        except errors.NoURL:
            self.assertFalse(self.fs.hasurl('foo'))
        else:
            self.assertTrue(self.fs.hasurl('foo'))
        self.fs.hasurl('a/b/c/foo/bar')

    def test_geturl_purpose(self):
        """Check an unknown purpose raises a NoURL error."""
        self.fs.create('foo')
        with self.assertRaises(errors.NoURL):
            self.fs.geturl('foo', purpose='__nosuchpurpose__')

    def test_validatepath(self):
        """Check validatepath returns an absolute path."""
        path = self.fs.validatepath('foo')
        self.assertEqual(path, '/foo')

    def test_invalid_chars(self):
        with self.assertRaises(errors.InvalidCharsInPath):
            self.fs.open('invalid\x00file', 'wb')
        with self.assertRaises(errors.InvalidCharsInPath):
            self.fs.validatepath('invalid\x00file')

    def test_getinfo(self):
        root_info = self.fs.getinfo('/')
        self.assertEqual(root_info.name, '')
        self.assertTrue(root_info.is_dir)
        self.assertIn('basic', root_info.namespaces)
        self.fs.writebytes('foo', b'bar')
        self.fs.makedir('dir')
        info = self.fs.getinfo('foo').raw
        self.assertIn('basic', info)
        self.assertIsInstance(info['basic']['name'], text_type)
        self.assertEqual(info['basic']['name'], 'foo')
        self.assertFalse(info['basic']['is_dir'])
        info = self.fs.getinfo('dir').raw
        self.assertIn('basic', info)
        self.assertEqual(info['basic']['name'], 'dir')
        self.assertTrue(info['basic']['is_dir'])
        info = self.fs.getinfo('foo', namespaces=['details']).raw
        self.assertIn('basic', info)
        self.assertIsInstance(info, dict)
        self.assertEqual(info['details']['size'], 3)
        self.assertEqual(info['details']['type'], int(ResourceType.file))
        self.assertEqual(info, self.fs.getdetails('foo').raw)
        try:
            json.dumps(info)
        except (TypeError, ValueError):
            raise AssertionError('info should be JSON serializable')
        no_info = self.fs.getinfo('foo', '__nosuchnamespace__').raw
        self.assertIsInstance(no_info, dict)
        self.assertEqual(no_info['basic'], {'name': 'foo', 'is_dir': False})
        info = self.fs.getinfo('foo', namespaces=['access', 'stat', 'details'])
        if 'details' in info.namespaces:
            details = info.raw['details']
            self.assertIsInstance(details.get('accessed'), (type(None), int, float))
            self.assertIsInstance(details.get('modified'), (type(None), int, float))
            self.assertIsInstance(details.get('created'), (type(None), int, float))
            self.assertIsInstance(details.get('metadata_changed'), (type(None), int, float))

    def test_exists(self):
        self.assertTrue(self.fs.exists('/'))
        self.assertTrue(self.fs.exists(''))
        self.assertFalse(self.fs.exists('foo'))
        self.assertFalse(self.fs.exists('foo/bar'))
        self.assertFalse(self.fs.exists('foo/bar/baz'))
        self.assertFalse(self.fs.exists('egg'))
        self.fs.makedirs('foo/bar')
        self.fs.writebytes('foo/bar/baz', b'test')
        self.assertTrue(self.fs.exists('foo'))
        self.assertTrue(self.fs.exists('foo/bar'))
        self.assertTrue(self.fs.exists('foo/bar/baz'))
        self.assertFalse(self.fs.exists('egg'))
        self.assert_exists('foo')
        self.assert_exists('foo/bar')
        self.assert_exists('foo/bar/baz')
        self.assert_not_exists('egg')
        self.fs.remove('foo/bar/baz')
        self.assert_not_exists('foo/bar/baz')
        self.assertFalse(self.fs.exists('foo/bar/baz'))
        self.assert_not_exists('foo/bar/baz')
        self.assertTrue(self.fs.exists('/'))
        self.assertTrue(self.fs.exists(''))

    def test_listdir(self):
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.listdir('foobar')
        self.assertEqual(self.fs.listdir('/'), [])
        self.assertEqual(self.fs.listdir('.'), [])
        self.assertEqual(self.fs.listdir('./'), [])
        self.fs.writebytes('foo', b'egg')
        self.fs.writebytes('bar', b'egg')
        self.fs.makedir('baz')
        self.fs.writebytes('baz/egg', b'egg')
        six.assertCountEqual(self, self.fs.listdir('/'), ['foo', 'bar', 'baz'])
        six.assertCountEqual(self, self.fs.listdir('.'), ['foo', 'bar', 'baz'])
        six.assertCountEqual(self, self.fs.listdir('./'), ['foo', 'bar', 'baz'])
        for name in self.fs.listdir('/'):
            self.assertIsInstance(name, text_type)
        self.fs.makedir('dir')
        self.assertEqual(self.fs.listdir('/dir'), [])
        self.fs.writebytes('dir/foofoo', b'egg')
        self.fs.writebytes('dir/barbar', b'egg')
        six.assertCountEqual(self, self.fs.listdir('dir'), ['foofoo', 'barbar'])
        for name in self.fs.listdir('dir'):
            self.assertIsInstance(name, text_type)
        self.fs.create('notadir')
        with self.assertRaises(errors.DirectoryExpected):
            self.fs.listdir('notadir')

    def test_move(self):
        self.fs.writebytes('foo', b'egg')
        self.assert_isfile('foo')
        self.fs.move('foo', 'bar')
        self.assert_not_exists('foo')
        self.assert_exists('bar')
        self.assert_bytes('bar', b'egg')
        self.fs.writebytes('foo2', b'eggegg')
        with self.assertRaises(errors.DestinationExists):
            self.fs.move('foo2', 'bar')
        self.fs.move('foo2', 'bar', overwrite=True)
        self.assert_not_exists('foo2')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.move('bar', 'egg/bar')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.move('egg', 'spam')
        self.fs.makedir('baz')
        self.fs.writebytes('baz/bazbaz', b'bazbaz')
        self.fs.makedir('baz2')
        self.fs.move('baz/bazbaz', 'baz2/bazbaz')
        self.assert_not_exists('baz/bazbaz')
        self.assert_bytes('baz2/bazbaz', b'bazbaz')
        self.assert_isdir('baz2')
        self.assert_not_exists('yolk')
        with self.assertRaises(errors.FileExpected):
            self.fs.move('baz2', 'yolk')

    def test_makedir(self):
        with self.assertRaises(errors.DirectoryExists):
            self.fs.makedir('/')
        slash_fs = self.fs.makedir('/', recreate=True)
        self.assertIsInstance(slash_fs, SubFS)
        self.assertEqual(self.fs.listdir('/'), [])
        self.assert_not_exists('foo')
        self.fs.makedir('foo')
        self.assert_isdir('foo')
        self.assertEqual(self.fs.gettype('foo'), ResourceType.directory)
        self.fs.writebytes('foo/bar.txt', b'egg')
        self.assert_bytes('foo/bar.txt', b'egg')
        with self.assertRaises(errors.DirectoryExists):
            self.fs.makedir('foo')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.makedir('/foo/bar/baz')
        self.fs.makedir('/foo/bar')
        self.fs.makedir('/foo/bar/baz')
        with self.assertRaises(errors.DirectoryExists):
            self.fs.makedir('foo/bar/baz')
        with self.assertRaises(errors.DirectoryExists):
            self.fs.makedir('foo/bar.txt')

    def test_makedirs(self):
        self.assertFalse(self.fs.exists('foo'))
        self.fs.makedirs('foo')
        self.assertEqual(self.fs.gettype('foo'), ResourceType.directory)
        self.fs.makedirs('foo/bar/baz')
        self.assertTrue(self.fs.isdir('foo/bar'))
        self.assertTrue(self.fs.isdir('foo/bar/baz'))
        with self.assertRaises(errors.DirectoryExists):
            self.fs.makedirs('foo/bar/baz')
        self.fs.makedirs('foo/bar/baz', recreate=True)
        self.fs.writebytes('foo.bin', b'test')
        with self.assertRaises(errors.DirectoryExpected):
            self.fs.makedirs('foo.bin/bar')
        with self.assertRaises(errors.DirectoryExpected):
            self.fs.makedirs('foo.bin/bar/baz/egg')

    def test_repeat_dir(self):
        self.fs.makedirs('foo/foo/foo')
        self.assertEqual(self.fs.listdir(''), ['foo'])
        self.assertEqual(self.fs.listdir('foo'), ['foo'])
        self.assertEqual(self.fs.listdir('foo/foo'), ['foo'])
        self.assertEqual(self.fs.listdir('foo/foo/foo'), [])
        scan = list(self.fs.scandir('foo'))
        self.assertEqual(len(scan), 1)
        self.assertEqual(scan[0].name, 'foo')

    def test_open(self):
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.open('doesnotexist', 'r')
        self.fs.makedir('foo')
        text = 'Hello, World'
        with self.fs.open('foo/hello', 'wt') as f:
            repr(f)
            self.assertIsInstance(f, io.IOBase)
            self.assertTrue(f.writable())
            self.assertFalse(f.readable())
            self.assertFalse(f.closed)
            f.write(text)
        self.assertTrue(f.closed)
        with self.fs.open('foo/hello', 'rt') as f:
            self.assertIsInstance(f, io.IOBase)
            self.assertTrue(f.readable())
            self.assertFalse(f.writable())
            self.assertFalse(f.closed)
            hello = f.read()
        self.assertTrue(f.closed)
        self.assertEqual(hello, text)
        self.assert_text('foo/hello', text)
        text = 'Goodbye, World'
        with self.fs.open('foo/hello', 'wt') as f:
            f.write(text)
        self.assert_text('foo/hello', text)
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.open('/foo/bar/test.txt')
        with self.fs.open('foo/hello') as f:
            try:
                fn = f.fileno()
            except io.UnsupportedOperation:
                pass
            else:
                self.assertEqual(os.read(fn, 7), b'Goodbye')
        lines = os.linesep.join(['Line 1', 'Line 2', 'Line 3'])
        self.fs.writetext('iter.txt', lines)
        with self.fs.open('iter.txt') as f:
            for actual, expected in zip(f, lines.splitlines(1)):
                self.assertEqual(actual, expected)

    def test_openbin_rw(self):
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.openbin('doesnotexist', 'r')
        self.fs.makedir('foo')
        text = b'Hello, World\n'
        with self.fs.openbin('foo/hello', 'w') as f:
            repr(f)
            self.assertIn('b', f.mode)
            self.assertIsInstance(f, io.IOBase)
            self.assertTrue(f.writable())
            self.assertFalse(f.readable())
            self.assertEqual(len(text), f.write(text))
            self.assertFalse(f.closed)
        self.assertTrue(f.closed)
        with self.assertRaises(errors.FileExists):
            with self.fs.openbin('foo/hello', 'x') as f:
                pass
        with self.fs.openbin('foo/hello', 'r') as f:
            self.assertIn('b', f.mode)
            self.assertIsInstance(f, io.IOBase)
            self.assertTrue(f.readable())
            self.assertFalse(f.writable())
            hello = f.read()
            self.assertFalse(f.closed)
        self.assertTrue(f.closed)
        self.assertEqual(hello, text)
        self.assert_bytes('foo/hello', text)
        text = b'Goodbye, World'
        with self.fs.openbin('foo/hello', 'w') as f:
            self.assertEqual(len(text), f.write(text))
        self.assert_bytes('foo/hello', text)
        with self.assertRaises(errors.FileExpected):
            self.fs.openbin('foo')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.openbin('/foo/bar/test.txt')
        with self.fs.openbin('foo/hello') as f:
            try:
                fn = f.fileno()
            except io.UnsupportedOperation:
                pass
            else:
                self.assertEqual(os.read(fn, 7), b'Goodbye')
        lines = b'\n'.join([b'Line 1', b'Line 2', b'Line 3'])
        self.fs.writebytes('iter.bin', lines)
        with self.fs.openbin('iter.bin') as f:
            for actual, expected in zip(f, lines.splitlines(1)):
                self.assertEqual(actual, expected)

    def test_open_files(self):
        with self.fs.open('text', 'w') as f:
            repr(f)
            text_type(f)
            self.assertIsInstance(f, io.IOBase)
            self.assertTrue(f.writable())
            self.assertFalse(f.readable())
            self.assertFalse(f.closed)
            self.assertEqual(f.tell(), 0)
            f.write('Hello\nWorld\n')
            self.assertEqual(f.tell(), 12)
            f.writelines(['foo\n', 'bar\n', 'baz\n'])
            with self.assertRaises(IOError):
                f.read(1)
        self.assertTrue(f.closed)
        with self.fs.open('bin', 'wb') as f:
            with self.assertRaises(IOError):
                f.read(1)
        with self.fs.open('text', 'r') as f:
            repr(f)
            text_type(f)
            self.assertIsInstance(f, io.IOBase)
            self.assertFalse(f.writable())
            self.assertTrue(f.readable())
            self.assertFalse(f.closed)
            self.assertEqual(f.readlines(), ['Hello\n', 'World\n', 'foo\n', 'bar\n', 'baz\n'])
            with self.assertRaises(IOError):
                f.write('no')
        self.assertTrue(f.closed)
        with self.fs.open('text', 'rb') as f:
            self.assertIsInstance(f, io.IOBase)
            self.assertFalse(f.writable())
            self.assertTrue(f.readable())
            self.assertFalse(f.closed)
            self.assertEqual(f.readlines(8), [b'Hello\n', b'World\n'])
            self.assertEqual(f.tell(), 12)
            buffer = bytearray(4)
            self.assertEqual(f.readinto(buffer), 4)
            self.assertEqual(f.tell(), 16)
            self.assertEqual(buffer, b'foo\n')
            with self.assertRaises(IOError):
                f.write(b'no')
        self.assertTrue(f.closed)
        with self.fs.open('text', 'r') as f:
            self.assertEqual(list(f), ['Hello\n', 'World\n', 'foo\n', 'bar\n', 'baz\n'])
            self.assertFalse(f.closed)
        self.assertTrue(f.closed)
        with self.fs.open('text') as f:
            iter_lines = iter(f)
            self.assertEqual(next(iter_lines), 'Hello\n')
        with self.fs.open('unicode', 'w') as f:
            self.assertEqual(12, f.write('Héllo\nWörld\n'))
        with self.fs.open('text', 'rb') as f:
            self.assertIsInstance(f, io.IOBase)
            self.assertFalse(f.writable())
            self.assertTrue(f.readable())
            self.assertTrue(f.seekable())
            self.assertFalse(f.closed)
            self.assertEqual(f.read(1), b'H')
            self.assertEqual(3, f.seek(3, Seek.set))
            self.assertEqual(f.read(1), b'l')
            self.assertEqual(6, f.seek(2, Seek.current))
            self.assertEqual(f.read(1), b'W')
            self.assertEqual(22, f.seek(-2, Seek.end))
            self.assertEqual(f.read(1), b'z')
            with self.assertRaises(ValueError):
                f.seek(10, 77)
        self.assertTrue(f.closed)
        with self.fs.open('text', 'r+b') as f:
            self.assertIsInstance(f, io.IOBase)
            self.assertTrue(f.readable())
            self.assertTrue(f.writable())
            self.assertTrue(f.seekable())
            self.assertFalse(f.closed)
            self.assertEqual(5, f.seek(5))
            self.assertEqual(5, f.truncate())
            self.assertEqual(0, f.seek(0))
            self.assertEqual(f.read(), b'Hello')
            self.assertEqual(10, f.truncate(10))
            self.assertEqual(5, f.tell())
            self.assertEqual(0, f.seek(0))
            print(repr(self.fs))
            print(repr(f))
            self.assertEqual(f.read(), b'Hello\x00\x00\x00\x00\x00')
            self.assertEqual(4, f.seek(4))
            f.write(b'O')
            self.assertEqual(4, f.seek(4))
            self.assertEqual(f.read(1), b'O')
        self.assertTrue(f.closed)

    def test_openbin(self):
        with self.fs.openbin('file.bin', 'wb') as write_file:
            repr(write_file)
            text_type(write_file)
            self.assertIn('b', write_file.mode)
            self.assertIsInstance(write_file, io.IOBase)
            self.assertTrue(write_file.writable())
            self.assertFalse(write_file.readable())
            self.assertFalse(write_file.closed)
            self.assertEqual(3, write_file.write(b'\x00\x01\x02'))
        self.assertTrue(write_file.closed)
        with self.fs.openbin('file.bin', 'rb') as read_file:
            repr(write_file)
            text_type(write_file)
            self.assertIn('b', read_file.mode)
            self.assertIsInstance(read_file, io.IOBase)
            self.assertTrue(read_file.readable())
            self.assertFalse(read_file.writable())
            self.assertFalse(read_file.closed)
            data = read_file.read()
        self.assertEqual(data, b'\x00\x01\x02')
        self.assertTrue(read_file.closed)
        with self.assertRaises(ValueError):
            with self.fs.openbin('file.bin', 'rt') as read_file:
                pass
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.openbin('foo.bin')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.openbin('/foo/bar/test.txt')
        self.fs.makedir('foo')
        with self.assertRaises(errors.FileExpected):
            self.fs.openbin('/foo')
        with self.assertRaises(errors.FileExpected):
            self.fs.openbin('/foo', 'w')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.openbin('/egg/bar')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.openbin('/egg/bar', 'w')
        with self.assertRaises(ValueError):
            self.fs.openbin('foo.bin', 'h')

    def test_open_exclusive(self):
        with self.fs.open('test_open_exclusive', 'x') as f:
            f.write('bananas')
        with self.assertRaises(errors.FileExists):
            self.fs.open('test_open_exclusive', 'x')

    def test_openbin_exclusive(self):
        with self.fs.openbin('test_openbin_exclusive', 'x') as f:
            f.write(b'bananas')
        with self.assertRaises(errors.FileExists):
            self.fs.openbin('test_openbin_exclusive', 'x')

    def test_opendir(self):
        self.fs.makedir('foo')
        self.fs.writebytes('foo/bar', b'barbar')
        self.fs.writebytes('foo/egg', b'eggegg')
        with self.fs.opendir('foo') as foo_fs:
            repr(foo_fs)
            text_type(foo_fs)
            six.assertCountEqual(self, foo_fs.listdir('/'), ['bar', 'egg'])
            self.assertTrue(foo_fs.isfile('bar'))
            self.assertTrue(foo_fs.isfile('egg'))
            self.assertEqual(foo_fs.readbytes('bar'), b'barbar')
            self.assertEqual(foo_fs.readbytes('egg'), b'eggegg')
        self.assertFalse(self.fs.isclosed())
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.opendir('egg')
        with self.assertRaises(errors.DirectoryExpected):
            self.fs.opendir('foo/egg')
        self.fs.opendir('')
        self.fs.opendir('/')
        with self.fs.opendir('foo', factory=ClosingSubFS) as foo_fs:
            six.assertCountEqual(self, foo_fs.listdir('/'), ['bar', 'egg'])
            self.assertTrue(foo_fs.isfile('bar'))
            self.assertTrue(foo_fs.isfile('egg'))
            self.assertEqual(foo_fs.readbytes('bar'), b'barbar')
            self.assertEqual(foo_fs.readbytes('egg'), b'eggegg')
        self.assertTrue(self.fs.isclosed())

    def test_remove(self):
        self.fs.writebytes('foo1', b'test1')
        self.fs.writebytes('foo2', b'test2')
        self.fs.writebytes('foo3', b'test3')
        self.assert_isfile('foo1')
        self.assert_isfile('foo2')
        self.assert_isfile('foo3')
        self.fs.remove('foo2')
        self.assert_isfile('foo1')
        self.assert_not_exists('foo2')
        self.assert_isfile('foo3')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.remove('bar')
        self.fs.makedir('dir')
        with self.assertRaises(errors.FileExpected):
            self.fs.remove('dir')
        self.fs.makedirs('foo/bar/baz/')
        error_msg = "resource 'foo/bar/egg/test.txt' not found"
        assertRaisesRegex = getattr(self, 'assertRaisesRegex', self.assertRaisesRegexp)
        with assertRaisesRegex(errors.ResourceNotFound, error_msg):
            self.fs.remove('foo/bar/egg/test.txt')

    def test_removedir(self):
        with self.assertRaises(errors.RemoveRootError):
            self.fs.removedir('/')
        self.fs.makedirs('foo/bar/baz')
        self.assertTrue(self.fs.exists('foo/bar/baz'))
        self.fs.removedir('foo/bar/baz')
        self.assertFalse(self.fs.exists('foo/bar/baz'))
        self.assertTrue(self.fs.isdir('foo/bar'))
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.removedir('nodir')
        self.fs.makedirs('foo/bar/baz')
        self.fs.writebytes('foo/egg', b'test')
        with self.assertRaises(errors.DirectoryExpected):
            self.fs.removedir('foo/egg')
        with self.assertRaises(errors.DirectoryNotEmpty):
            self.fs.removedir('foo/bar')

    def test_removetree(self):
        self.fs.makedirs('spam')
        self.fs.makedirs('foo/bar/baz')
        self.fs.makedirs('foo/egg')
        self.fs.makedirs('foo/a/b/c/d/e')
        self.fs.create('foo/egg.txt')
        self.fs.create('foo/bar/egg.bin')
        self.fs.create('foo/bar/baz/egg.txt')
        self.fs.create('foo/a/b/c/1.txt')
        self.fs.create('foo/a/b/c/2.txt')
        self.fs.create('foo/a/b/c/3.txt')
        self.assert_exists('foo/egg.txt')
        self.assert_exists('foo/bar/egg.bin')
        self.fs.removetree('foo')
        self.assert_not_exists('foo')
        self.assert_exists('spam')
        self.fs.create('bar')
        with self.assertRaises(errors.DirectoryExpected):
            self.fs.removetree('bar')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.removetree('foofoo')

    def test_removetree_root(self):
        self.fs.makedirs('foo/bar/baz')
        self.fs.makedirs('foo/egg')
        self.fs.makedirs('foo/a/b/c/d/e')
        self.fs.create('foo/egg.txt')
        self.fs.create('foo/bar/egg.bin')
        self.fs.create('foo/a/b/c/1.txt')
        self.fs.create('foo/a/b/c/2.txt')
        self.fs.create('foo/a/b/c/3.txt')
        self.assert_exists('foo/egg.txt')
        self.assert_exists('foo/bar/egg.bin')
        self.fs.removetree('/')
        self.assert_exists('/')
        self.assert_isempty('/')
        self.fs.create('egg')
        self.fs.makedir('yolk')
        self.assert_exists('egg')
        self.assert_exists('yolk')

    def test_setinfo(self):
        self.fs.create('birthday.txt')
        now = time.time()
        change_info = {'details': {'accessed': now + 60, 'modified': now + 60 * 60}}
        self.fs.setinfo('birthday.txt', change_info)
        new_info = self.fs.getinfo('birthday.txt', namespaces=['details'])
        can_write_acccess = new_info.is_writeable('details', 'accessed')
        can_write_modified = new_info.is_writeable('details', 'modified')
        if can_write_acccess:
            self.assertAlmostEqual(new_info.get('details', 'accessed'), now + 60, places=4)
        if can_write_modified:
            self.assertAlmostEqual(new_info.get('details', 'modified'), now + 60 * 60, places=4)
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.setinfo('nothing', {})

    def test_settimes(self):
        self.fs.create('birthday.txt')
        self.fs.settimes('birthday.txt', accessed=datetime(2016, 7, 5))
        info = self.fs.getinfo('birthday.txt', namespaces=['details'])
        can_write_acccess = info.is_writeable('details', 'accessed')
        can_write_modified = info.is_writeable('details', 'modified')
        if can_write_acccess:
            self.assertEqual(info.accessed, datetime(2016, 7, 5, tzinfo=timezone.utc))
        if can_write_modified:
            self.assertEqual(info.modified, datetime(2016, 7, 5, tzinfo=timezone.utc))

    def test_touch(self):
        self.fs.touch('new.txt')
        self.assert_isfile('new.txt')
        self.fs.settimes('new.txt', datetime(2016, 7, 5))
        info = self.fs.getinfo('new.txt', namespaces=['details'])
        if info.is_writeable('details', 'accessed'):
            self.assertEqual(info.accessed, datetime(2016, 7, 5, tzinfo=timezone.utc))
            now = time.time()
            self.fs.touch('new.txt')
            accessed = self.fs.getinfo('new.txt', namespaces=['details']).raw['details']['accessed']
            self.assertTrue(accessed - now < 5)

    def test_close(self):
        self.assertFalse(self.fs.isclosed())
        self.fs.close()
        self.assertTrue(self.fs.isclosed())
        self.fs.close()
        self.assertTrue(self.fs.isclosed())
        with self.assertRaises(errors.FilesystemClosed):
            self.fs.openbin('test.bin')

    def test_copy(self):
        self.fs.writebytes('foo', b'test')
        self.fs.copy('foo', 'bar')
        self.assert_bytes('bar', b'test')
        self.fs.writebytes('baz', b'truncateme')
        self.fs.copy('foo', 'baz', overwrite=True)
        self.assert_bytes('foo', b'test')
        with self.assertRaises(errors.DestinationExists):
            self.fs.copy('baz', 'foo')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.copy('baz', 'a/b/c/baz')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.copy('egg', 'spam')
        self.fs.makedir('dir')
        with self.assertRaises(errors.FileExpected):
            self.fs.copy('dir', 'folder')

    def _test_upload(self, workers):
        """Test fs.copy with varying number of worker threads."""
        with open_fs('temp://') as src_fs:
            src_fs.writebytes('foo', self.data1)
            src_fs.writebytes('bar', self.data2)
            src_fs.makedir('dir1').writebytes('baz', self.data3)
            src_fs.makedirs('dir2/dir3').writebytes('egg', self.data4)
            dst_fs = self.fs
            fs.copy.copy_fs(src_fs, dst_fs, workers=workers)
            self.assertEqual(dst_fs.readbytes('foo'), self.data1)
            self.assertEqual(dst_fs.readbytes('bar'), self.data2)
            self.assertEqual(dst_fs.readbytes('dir1/baz'), self.data3)
            self.assertEqual(dst_fs.readbytes('dir2/dir3/egg'), self.data4)

    def test_upload_0(self):
        self._test_upload(0)

    def test_upload_1(self):
        self._test_upload(1)

    def test_upload_2(self):
        self._test_upload(2)

    def test_upload_4(self):
        self._test_upload(4)

    def _test_download(self, workers):
        """Test fs.copy with varying number of worker threads."""
        src_fs = self.fs
        with open_fs('temp://') as dst_fs:
            src_fs.writebytes('foo', self.data1)
            src_fs.writebytes('bar', self.data2)
            src_fs.makedir('dir1').writebytes('baz', self.data3)
            src_fs.makedirs('dir2/dir3').writebytes('egg', self.data4)
            fs.copy.copy_fs(src_fs, dst_fs, workers=workers)
            self.assertEqual(dst_fs.readbytes('foo'), self.data1)
            self.assertEqual(dst_fs.readbytes('bar'), self.data2)
            self.assertEqual(dst_fs.readbytes('dir1/baz'), self.data3)
            self.assertEqual(dst_fs.readbytes('dir2/dir3/egg'), self.data4)

    def test_download_0(self):
        self._test_download(0)

    def test_download_1(self):
        self._test_download(1)

    def test_download_2(self):
        self._test_download(2)

    def test_download_4(self):
        self._test_download(4)

    def test_create(self):
        self.assertFalse(self.fs.exists('foo'))
        self.fs.create('foo')
        self.assertTrue(self.fs.exists('foo'))
        self.assertEqual(self.fs.gettype('foo'), ResourceType.file)
        self.assertEqual(self.fs.getsize('foo'), 0)
        self.fs.writebytes('foo', b'bar')
        self.assertEqual(self.fs.getsize('foo'), 3)
        self.fs.create('foo', wipe=True)
        self.assertEqual(self.fs.getsize('foo'), 0)
        self.fs.writebytes('foo', b'bar')
        self.assertEqual(self.fs.getsize('foo'), 3)
        self.fs.create('foo', wipe=False)
        self.assertEqual(self.fs.getsize('foo'), 3)

    def test_desc(self):
        self.fs.create('foo')
        description = self.fs.desc('foo')
        self.assertIsInstance(description, text_type)
        self.fs.makedir('dir')
        self.fs.desc('dir')
        self.fs.desc('/')
        self.fs.desc('')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.desc('bar')

    def test_scandir(self):
        with self.assertRaises(errors.ResourceNotFound):
            for _info in self.fs.scandir('/foobar'):
                pass
        iter_scandir = self.fs.scandir('/')
        self.assertTrue(isinstance(iter_scandir, collections_abc.Iterable))
        self.assertEqual(list(iter_scandir), [])
        self.fs.create('foo')
        with self.assertRaises(errors.DirectoryExpected):
            list(self.fs.scandir('foo'))
        self.fs.create('bar')
        self.fs.makedir('dir')
        iter_scandir = self.fs.scandir('/')
        self.assertTrue(isinstance(iter_scandir, collections_abc.Iterable))
        scandir = sorted((r.raw for r in iter_scandir), key=lambda info: info['basic']['name'])
        scandir = [{'basic': i['basic']} for i in scandir]
        self.assertEqual(scandir, [{'basic': {'name': 'bar', 'is_dir': False}}, {'basic': {'name': 'dir', 'is_dir': True}}, {'basic': {'name': 'foo', 'is_dir': False}}])
        list(self.fs.scandir('/', namespaces=['details', 'link', 'stat', 'lstat', 'access']))
        page1 = list(self.fs.scandir('/', page=(None, 2)))
        self.assertEqual(len(page1), 2)
        page2 = list(self.fs.scandir('/', page=(2, 4)))
        self.assertEqual(len(page2), 1)
        page3 = list(self.fs.scandir('/', page=(4, 6)))
        self.assertEqual(len(page3), 0)
        paged = {r.name for r in itertools.chain(page1, page2)}
        self.assertEqual(paged, {'foo', 'bar', 'dir'})

    def test_filterdir(self):
        self.assertEqual(list(self.fs.filterdir('/', files=['*.py'])), [])
        self.fs.makedir('bar')
        self.fs.create('foo.txt')
        self.fs.create('foo.py')
        self.fs.create('foo.pyc')
        page1 = list(self.fs.filterdir('/', page=(None, 2)))
        page2 = list(self.fs.filterdir('/', page=(2, 4)))
        page3 = list(self.fs.filterdir('/', page=(4, 6)))
        self.assertEqual(len(page1), 2)
        self.assertEqual(len(page2), 2)
        self.assertEqual(len(page3), 0)
        names = [info.name for info in itertools.chain(page1, page2, page3)]
        self.assertEqual(set(names), {'foo.txt', 'foo.py', 'foo.pyc', 'bar'})
        dir_list = [info.name for info in self.fs.filterdir('/', files=['*.py'])]
        self.assertEqual(set(dir_list), {'bar', 'foo.py'})
        dir_list = [info.name for info in self.fs.filterdir('/', files=['*.py', '*.pyc'])]
        self.assertEqual(set(dir_list), {'bar', 'foo.py', 'foo.pyc'})
        dir_list = [info.name for info in self.fs.filterdir('/', exclude_dirs=['*'], files=['*.py', '*.pyc'])]
        self.assertEqual(set(dir_list), {'foo.py', 'foo.pyc'})
        dir_list = [info.name for info in self.fs.filterdir('/', exclude_files=['*'])]
        self.assertEqual(set(dir_list), {'bar'})
        with self.assertRaises(TypeError):
            dir_list = [info.name for info in self.fs.filterdir('/', files='*.py')]
        self.fs.makedir('baz')
        dir_list = [info.name for info in self.fs.filterdir('/', exclude_files=['*'], dirs=['??z'])]
        self.assertEqual(set(dir_list), {'baz'})
        with self.assertRaises(TypeError):
            dir_list = [info.name for info in self.fs.filterdir('/', exclude_files=['*'], dirs='*.py')]

    def test_readbytes(self):
        all_bytes = b''.join((six.int2byte(n) for n in range(256)))
        with self.fs.open('foo', 'wb') as f:
            f.write(all_bytes)
        self.assertEqual(self.fs.readbytes('foo'), all_bytes)
        _all_bytes = self.fs.readbytes('foo')
        self.assertIsInstance(_all_bytes, bytes)
        self.assertEqual(_all_bytes, all_bytes)
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.readbytes('foo/bar')
        self.fs.makedir('baz')
        with self.assertRaises(errors.FileExpected):
            self.fs.readbytes('baz')

    def test_download(self):
        test_bytes = b'Hello, World'
        self.fs.writebytes('hello.bin', test_bytes)
        write_file = io.BytesIO()
        self.fs.download('hello.bin', write_file)
        self.assertEqual(write_file.getvalue(), test_bytes)
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.download('foo.bin', write_file)

    def test_download_chunk_size(self):
        test_bytes = b'Hello, World' * 100
        self.fs.writebytes('hello.bin', test_bytes)
        write_file = io.BytesIO()
        self.fs.download('hello.bin', write_file, chunk_size=8)
        self.assertEqual(write_file.getvalue(), test_bytes)

    def test_isempty(self):
        self.assertTrue(self.fs.isempty('/'))
        self.fs.makedir('foo')
        self.assertFalse(self.fs.isempty('/'))
        self.assertTrue(self.fs.isempty('/foo'))
        self.fs.create('foo/bar.txt')
        self.assertFalse(self.fs.isempty('/foo'))
        self.fs.remove('foo/bar.txt')
        self.assertTrue(self.fs.isempty('/foo'))

    def test_writebytes(self):
        all_bytes = b''.join((six.int2byte(n) for n in range(256)))
        self.fs.writebytes('foo', all_bytes)
        with self.fs.open('foo', 'rb') as f:
            _bytes = f.read()
        self.assertIsInstance(_bytes, bytes)
        self.assertEqual(_bytes, all_bytes)
        self.assert_bytes('foo', all_bytes)
        with self.assertRaises(TypeError):
            self.fs.writebytes('notbytes', 'unicode')

    def test_readtext(self):
        self.fs.makedir('foo')
        with self.fs.open('foo/unicode.txt', 'wt') as f:
            f.write(UNICODE_TEXT)
        text = self.fs.readtext('foo/unicode.txt')
        self.assertIsInstance(text, text_type)
        self.assertEqual(text, UNICODE_TEXT)
        self.assert_text('foo/unicode.txt', UNICODE_TEXT)

    def test_writetext(self):
        self.fs.writetext('foo', 'bar')
        with self.fs.open('foo', 'rt') as f:
            foo = f.read()
        self.assertEqual(foo, 'bar')
        self.assertIsInstance(foo, text_type)
        with self.assertRaises(TypeError):
            self.fs.writetext('nottext', b'bytes')

    def test_writefile(self):
        bytes_file = io.BytesIO(b'bar')
        self.fs.writefile('foo', bytes_file)
        with self.fs.open('foo', 'rb') as f:
            data = f.read()
        self.assertEqual(data, b'bar')

    def test_upload(self):
        bytes_file = io.BytesIO(b'bar')
        self.fs.upload('foo', bytes_file)
        with self.fs.open('foo', 'rb') as f:
            data = f.read()
        self.assertEqual(data, b'bar')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.upload('/spam/eggs', bytes_file)

    def test_upload_chunk_size(self):
        test_data = b'bar' * 128
        bytes_file = io.BytesIO(test_data)
        self.fs.upload('foo', bytes_file, chunk_size=8)
        with self.fs.open('foo', 'rb') as f:
            data = f.read()
        self.assertEqual(data, test_data)

    def test_bin_files(self):
        with self.fs.openbin('foo1', 'wb') as f:
            text_type(f)
            repr(f)
            f.write(b'a')
            f.write(b'b')
            f.write(b'c')
        self.assert_bytes('foo1', b'abc')
        with self.fs.openbin('foo2', 'wb') as f:
            f.writelines([b'hello\n', b'world'])
        self.assert_bytes('foo2', b'hello\nworld')
        with self.fs.openbin('foo2') as f:
            self.assertEqual(f.readline(), b'hello\n')
            self.assertEqual(f.readline(), b'world')
        with self.fs.openbin('foo2') as f:
            lines = f.readlines()
        self.assertEqual(lines, [b'hello\n', b'world'])
        with self.fs.openbin('foo2') as f:
            lines = list(f)
        self.assertEqual(lines, [b'hello\n', b'world'])
        with self.fs.openbin('foo2') as f:
            lines = []
            for line in f:
                lines.append(line)
        self.assertEqual(lines, [b'hello\n', b'world'])
        with self.fs.openbin('foo2') as f:
            print(repr(f))
            self.assertEqual(next(f), b'hello\n')
        with self.fs.open('foo2', 'r+b') as f:
            f.truncate(3)
        self.assertEqual(self.fs.getsize('foo2'), 3)
        self.assert_bytes('foo2', b'hel')

    def test_files(self):
        with self.fs.open('foo1', 'wt') as f:
            text_type(f)
            repr(f)
            f.write('a')
            f.write('b')
            f.write('c')
        self.assert_text('foo1', 'abc')
        with self.fs.open('foo2', 'wt') as f:
            f.writelines(['hello\n', 'world'])
        self.assert_text('foo2', 'hello\nworld')
        with self.fs.open('foo2') as f:
            self.assertEqual(f.readline(), 'hello\n')
            self.assertEqual(f.readline(), 'world')
        with self.fs.open('foo2') as f:
            lines = f.readlines()
        self.assertEqual(lines, ['hello\n', 'world'])
        with self.fs.open('foo2') as f:
            lines = list(f)
        self.assertEqual(lines, ['hello\n', 'world'])
        with self.fs.open('foo2') as f:
            lines = []
            for line in f:
                lines.append(line)
        self.assertEqual(lines, ['hello\n', 'world'])
        with self.fs.open('foo2', 'r+') as f:
            f.truncate(3)
        self.assertEqual(self.fs.getsize('foo2'), 3)
        self.assert_text('foo2', 'hel')
        with self.fs.open('foo2', 'ab') as f:
            f.write(b'p')
        self.assert_bytes('foo2', b'help')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = self.fs.open('foo2', 'r')
            del f
        with self.assertRaises(IOError):
            with self.fs.open('foo2', 'r') as f:
                f.write('no!')
        with self.assertRaises(IOError):
            with self.fs.open('newfoo', 'w') as f:
                f.read(2)

    def test_copy_file(self):
        bytes_test = b'Hello, World'
        self.fs.writebytes('foo.txt', bytes_test)
        fs.copy.copy_file(self.fs, 'foo.txt', self.fs, 'bar.txt')
        self.assert_bytes('bar.txt', bytes_test)
        mem_fs = open_fs('mem://')
        fs.copy.copy_file(self.fs, 'foo.txt', mem_fs, 'bar.txt')
        self.assertEqual(mem_fs.readbytes('bar.txt'), bytes_test)

    def test_copy_structure(self):
        mem_fs = open_fs('mem://')
        self.fs.makedirs('foo/bar/baz')
        self.fs.makedir('egg')
        fs.copy.copy_structure(self.fs, mem_fs)
        expected = {'/egg', '/foo', '/foo/bar', '/foo/bar/baz'}
        self.assertEqual(set(walk.walk_dirs(mem_fs)), expected)

    def _test_copy_dir(self, protocol):
        other_fs = open_fs(protocol)
        self.fs.makedirs('foo/bar/baz')
        self.fs.makedir('egg')
        self.fs.writetext('top.txt', 'Hello, World')
        self.fs.writetext('/foo/bar/baz/test.txt', 'Goodbye, World')
        fs.copy.copy_dir(self.fs, '/', other_fs, '/')
        expected = {'/egg', '/foo', '/foo/bar', '/foo/bar/baz'}
        self.assertEqual(set(walk.walk_dirs(other_fs)), expected)
        self.assert_text('top.txt', 'Hello, World')
        self.assert_text('/foo/bar/baz/test.txt', 'Goodbye, World')
        other_fs = open_fs('mem://')
        fs.copy.copy_dir(self.fs, '/foo', other_fs, '/')
        self.assertEqual(list(walk.walk_files(other_fs)), ['/bar/baz/test.txt'])
        print('BEFORE')
        self.fs.tree()
        other_fs.tree()
        fs.copy.copy_dir(self.fs, '/foo', other_fs, '/egg')
        print('FS')
        self.fs.tree()
        print('OTHER')
        other_fs.tree()
        self.assertEqual(list(walk.walk_files(other_fs)), ['/bar/baz/test.txt', '/egg/bar/baz/test.txt'])

    def _test_copy_dir_write(self, protocol):
        other_fs = open_fs(protocol)
        other_fs.makedirs('foo/bar/baz')
        other_fs.makedir('egg')
        other_fs.writetext('top.txt', 'Hello, World')
        other_fs.writetext('/foo/bar/baz/test.txt', 'Goodbye, World')
        fs.copy.copy_dir(other_fs, '/', self.fs, '/')
        expected = {'/egg', '/foo', '/foo/bar', '/foo/bar/baz'}
        self.assertEqual(set(walk.walk_dirs(self.fs)), expected)
        self.assert_text('top.txt', 'Hello, World')
        self.assert_text('/foo/bar/baz/test.txt', 'Goodbye, World')

    def test_copy_dir_mem(self):
        self._test_copy_dir('mem://')
        self._test_copy_dir_write('mem://')

    def test_copy_dir_temp(self):
        self._test_copy_dir('temp://')
        self._test_copy_dir_write('temp://')

    def test_move_dir_same_fs(self):
        self.fs.makedirs('foo/bar/baz')
        self.fs.makedir('egg')
        self.fs.writetext('top.txt', 'Hello, World')
        self.fs.writetext('/foo/bar/baz/test.txt', 'Goodbye, World')
        fs.move.move_dir(self.fs, 'foo', self.fs, 'foo2')
        expected = {'/egg', '/foo2', '/foo2/bar', '/foo2/bar/baz'}
        self.assertEqual(set(walk.walk_dirs(self.fs)), expected)
        self.assert_text('top.txt', 'Hello, World')
        self.assert_text('/foo2/bar/baz/test.txt', 'Goodbye, World')
        self.assertEqual(sorted(self.fs.listdir('/')), ['egg', 'foo2', 'top.txt'])
        self.assertEqual(sorted((x.name for x in self.fs.scandir('/'))), ['egg', 'foo2', 'top.txt'])

    def _test_move_dir_write(self, protocol):
        other_fs = open_fs(protocol)
        other_fs.makedirs('foo/bar/baz')
        other_fs.makedir('egg')
        other_fs.writetext('top.txt', 'Hello, World')
        other_fs.writetext('/foo/bar/baz/test.txt', 'Goodbye, World')
        fs.move.move_dir(other_fs, '/', self.fs, '/')
        expected = {'/egg', '/foo', '/foo/bar', '/foo/bar/baz'}
        self.assertEqual(other_fs.listdir('/'), [])
        self.assertEqual(set(walk.walk_dirs(self.fs)), expected)
        self.assert_text('top.txt', 'Hello, World')
        self.assert_text('/foo/bar/baz/test.txt', 'Goodbye, World')

    def test_move_dir_mem(self):
        self._test_move_dir_write('mem://')

    def test_move_dir_temp(self):
        self._test_move_dir_write('temp://')

    def test_move_file_same_fs(self):
        text = 'Hello, World'
        self.fs.makedir('foo').writetext('test.txt', text)
        self.assert_text('foo/test.txt', text)
        fs.move.move_file(self.fs, 'foo/test.txt', self.fs, 'foo/test2.txt')
        self.assert_not_exists('foo/test.txt')
        self.assert_text('foo/test2.txt', text)
        self.assertEqual(self.fs.listdir('foo'), ['test2.txt'])
        self.assertEqual(next(self.fs.scandir('foo')).name, 'test2.txt')

    def _test_move_file(self, protocol):
        other_fs = open_fs(protocol)
        text = 'Hello, World'
        self.fs.makedir('foo').writetext('test.txt', text)
        self.assert_text('foo/test.txt', text)
        with self.assertRaises(errors.ResourceNotFound):
            fs.move.move_file(self.fs, 'foo/test.txt', other_fs, 'foo/test2.txt')
        other_fs.makedir('foo')
        fs.move.move_file(self.fs, 'foo/test.txt', other_fs, 'foo/test2.txt')
        self.assertEqual(other_fs.readtext('foo/test2.txt'), text)

    def test_move_file_mem(self):
        self._test_move_file('mem://')

    def test_move_file_temp(self):
        self._test_move_file('temp://')

    def test_copydir(self):
        self.fs.makedirs('foo/bar/baz/egg')
        self.fs.writetext('foo/bar/foofoo.txt', 'Hello')
        self.fs.makedir('foo2')
        self.fs.copydir('foo/bar', 'foo2')
        self.assert_text('foo2/foofoo.txt', 'Hello')
        self.assert_isdir('foo2/baz/egg')
        self.assert_text('foo/bar/foofoo.txt', 'Hello')
        self.assert_isdir('foo/bar/baz/egg')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.copydir('foo', 'foofoo')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.copydir('spam', 'egg', create=True)
        with self.assertRaises(errors.DirectoryExpected):
            self.fs.copydir('foo2/foofoo.txt', 'foofoo.txt', create=True)

    def test_movedir(self):
        self.fs.makedirs('foo/bar/baz/egg')
        self.fs.writetext('foo/bar/foofoo.txt', 'Hello')
        self.fs.makedir('foo2')
        self.fs.movedir('foo/bar', 'foo2')
        self.assert_text('foo2/foofoo.txt', 'Hello')
        self.assert_isdir('foo2/baz/egg')
        self.assert_not_exists('foo/bar')
        self.assert_not_exists('foo/bar/foofoo.txt')
        self.assert_not_exists('foo/bar/baz/egg')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.movedir('foo', 'foofoo')
        with self.assertRaises(errors.ResourceNotFound):
            self.fs.movedir('spam', 'egg', create=True)
        with self.assertRaises(errors.DirectoryExpected):
            self.fs.movedir('foo2/foofoo.txt', 'foo2/baz/egg')

    def test_match(self):
        self.assertTrue(self.fs.match(['*.py'], 'foo.py'))
        self.assertEqual(self.fs.match(['*.py'], 'FOO.PY'), self.fs.getmeta().get('case_insensitive', False))

    def test_tree(self):
        self.fs.makedirs('foo/bar')
        self.fs.create('test.txt')
        write_tree = io.StringIO()
        self.fs.tree(file=write_tree)
        written = write_tree.getvalue()
        expected = '|-- foo\n|   `-- bar\n`-- test.txt\n'
        self.assertEqual(expected, written)

    def test_unicode_path(self):
        if not self.fs.getmeta().get('unicode_paths', False):
            raise unittest.SkipTest('the filesystem does not support unicode paths.')
        self.fs.makedir('földér')
        self.fs.writetext('☭.txt', 'Smells like communism.')
        self.fs.writebytes('földér/☣.txt', b'Smells like an old syringe.')
        self.assert_isdir('földér')
        self.assertEqual(['☣.txt'], self.fs.listdir('földér'))
        self.assertEqual('☣.txt', self.fs.getinfo('földér/☣.txt').name)
        self.assert_text('☭.txt', 'Smells like communism.')
        self.assert_bytes('földér/☣.txt', b'Smells like an old syringe.')
        if self.fs.hassyspath('földér/☣.txt'):
            self.assertTrue(os.path.exists(self.fs.getsyspath('földér/☣.txt')))
        self.fs.remove('földér/☣.txt')
        self.assert_not_exists('földér/☣.txt')
        self.fs.removedir('földér')
        self.assert_not_exists('földér')

    def test_case_sensitive(self):
        meta = self.fs.getmeta()
        if 'case_insensitive' not in meta:
            raise unittest.SkipTest('case sensitivity not known')
        if meta.get('case_insensitive', False):
            raise unittest.SkipTest('the filesystem is not case sensitive.')
        self.fs.makedir('foo')
        self.fs.makedir('Foo')
        self.fs.touch('fOO')
        self.assert_exists('foo')
        self.assert_exists('Foo')
        self.assert_exists('fOO')
        self.assert_not_exists('FoO')
        self.assert_isdir('foo')
        self.assert_isdir('Foo')
        self.assert_isfile('fOO')

    def test_glob(self):
        self.assertIsInstance(self.fs.glob, glob.BoundGlobber)

    def test_hash(self):
        self.fs.makedir('foo').writebytes('hashme.txt', b'foobar' * 1024)
        self.assertEqual(self.fs.hash('foo/hashme.txt', 'md5'), '9fff4bb103ab8ce4619064109c54cb9c')
        with self.assertRaises(errors.UnsupportedHash):
            self.fs.hash('foo/hashme.txt', 'nohash')
        with self.fs.opendir('foo') as foo_fs:
            self.assertEqual(foo_fs.hash('hashme.txt', 'md5'), '9fff4bb103ab8ce4619064109c54cb9c')