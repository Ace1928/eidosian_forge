import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
class TestMemoryTransport(tests.TestCase):

    def test_get_transport(self):
        memory.MemoryTransport()

    def test_clone(self):
        t = memory.MemoryTransport()
        self.assertTrue(isinstance(t, memory.MemoryTransport))
        self.assertEqual('memory:///', t.clone('/').base)

    def test_abspath(self):
        t = memory.MemoryTransport()
        self.assertEqual('memory:///relpath', t.abspath('relpath'))

    def test_abspath_of_root(self):
        t = memory.MemoryTransport()
        self.assertEqual('memory:///', t.base)
        self.assertEqual('memory:///', t.abspath('/'))

    def test_abspath_of_relpath_starting_at_root(self):
        t = memory.MemoryTransport()
        self.assertEqual('memory:///foo', t.abspath('/foo'))

    def test_append_and_get(self):
        t = memory.MemoryTransport()
        t.append_bytes('path', b'content')
        self.assertEqual(t.get('path').read(), b'content')
        t.append_file('path', BytesIO(b'content'))
        with t.get('path') as f:
            self.assertEqual(f.read(), b'contentcontent')

    def test_put_and_get(self):
        t = memory.MemoryTransport()
        t.put_file('path', BytesIO(b'content'))
        self.assertEqual(t.get('path').read(), b'content')
        t.put_bytes('path', b'content')
        self.assertEqual(t.get('path').read(), b'content')

    def test_append_without_dir_fails(self):
        t = memory.MemoryTransport()
        self.assertRaises(NoSuchFile, t.append_bytes, 'dir/path', b'content')

    def test_put_without_dir_fails(self):
        t = memory.MemoryTransport()
        self.assertRaises(NoSuchFile, t.put_file, 'dir/path', BytesIO(b'content'))

    def test_get_missing(self):
        transport = memory.MemoryTransport()
        self.assertRaises(NoSuchFile, transport.get, 'foo')

    def test_has_missing(self):
        t = memory.MemoryTransport()
        self.assertEqual(False, t.has('foo'))

    def test_has_present(self):
        t = memory.MemoryTransport()
        t.append_bytes('foo', b'content')
        self.assertEqual(True, t.has('foo'))

    def test_list_dir(self):
        t = memory.MemoryTransport()
        t.put_bytes('foo', b'content')
        t.mkdir('dir')
        t.put_bytes('dir/subfoo', b'content')
        t.put_bytes('dirlike', b'content')
        self.assertEqual(['dir', 'dirlike', 'foo'], sorted(t.list_dir('.')))
        self.assertEqual(['subfoo'], sorted(t.list_dir('dir')))

    def test_mkdir(self):
        t = memory.MemoryTransport()
        t.mkdir('dir')
        t.append_bytes('dir/path', b'content')
        with t.get('dir/path') as f:
            self.assertEqual(f.read(), b'content')

    def test_mkdir_missing_parent(self):
        t = memory.MemoryTransport()
        self.assertRaises(NoSuchFile, t.mkdir, 'dir/dir')

    def test_mkdir_twice(self):
        t = memory.MemoryTransport()
        t.mkdir('dir')
        self.assertRaises(FileExists, t.mkdir, 'dir')

    def test_parameters(self):
        t = memory.MemoryTransport()
        self.assertEqual(True, t.listable())
        self.assertEqual(False, t.is_readonly())

    def test_iter_files_recursive(self):
        t = memory.MemoryTransport()
        t.mkdir('dir')
        t.put_bytes('dir/foo', b'content')
        t.put_bytes('dir/bar', b'content')
        t.put_bytes('bar', b'content')
        paths = set(t.iter_files_recursive())
        self.assertEqual({'dir/foo', 'dir/bar', 'bar'}, paths)

    def test_stat(self):
        t = memory.MemoryTransport()
        t.put_bytes('foo', b'content')
        t.put_bytes('bar', b'phowar')
        self.assertEqual(7, t.stat('foo').st_size)
        self.assertEqual(6, t.stat('bar').st_size)