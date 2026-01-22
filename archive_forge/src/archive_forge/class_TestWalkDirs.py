import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestWalkDirs(tests.TestCaseInTempDir):

    def assertExpectedBlocks(self, expected, result):
        self.assertEqual(expected, [(dirinfo, [line[0:3] for line in block]) for dirinfo, block in result])

    def test_walkdirs(self):
        tree = ['.bzr', '0file', '1dir/', '1dir/0file', '1dir/1dir/', '2file']
        self.build_tree(tree)
        expected_dirblocks = [(('', '.'), [('0file', '0file', 'file'), ('1dir', '1dir', 'directory'), ('2file', '2file', 'file')]), (('1dir', './1dir'), [('1dir/0file', '0file', 'file'), ('1dir/1dir', '1dir', 'directory')]), (('1dir/1dir', './1dir/1dir'), [])]
        result = []
        found_bzrdir = False
        for dirdetail, dirblock in osutils.walkdirs('.'):
            if len(dirblock) and dirblock[0][1] == '.bzr':
                found_bzrdir = True
                del dirblock[0]
            result.append((dirdetail, dirblock))
        self.assertTrue(found_bzrdir)
        self.assertExpectedBlocks(expected_dirblocks, result)
        result = []
        for dirblock in osutils.walkdirs('./1dir', '1dir'):
            result.append(dirblock)
        self.assertExpectedBlocks(expected_dirblocks[1:], result)

    def test_walkdirs_os_error(self):
        if sys.platform == 'win32':
            raise tests.TestNotApplicable('readdir IOError not tested on win32')
        self.requireFeature(features.not_running_as_root)
        os.mkdir('test-unreadable')
        os.chmod('test-unreadable', 0)
        self.addCleanup(os.chmod, 'test-unreadable', 448)
        e = self.assertRaises(OSError, list, osutils._walkdirs_utf8('.'))
        self.assertEqual('./test-unreadable', osutils.safe_unicode(e.filename))
        self.assertEqual(errno.EACCES, e.errno)
        self.assertContainsRe(str(e), '\\./test-unreadable')

    def test_walkdirs_encoding_error(self):
        self.requireFeature(features.ByteStringNamedFilesystem)
        tree = ['.bzr', '0file', '1dir/', '1dir/0file', '1dir/1dir/', '1file']
        self.build_tree(tree)
        os.rename(b'./1file', b'\xe8file')
        if b'\xe8file' not in os.listdir('.'):
            self.skipTest('Lack filesystem that preserves arbitrary bytes')
        self._save_platform_info()

        def attempt():
            for dirdetail, dirblock in osutils.walkdirs(b'.', codecs.utf_8_decode):
                pass
        self.assertRaises(UnicodeDecodeError, attempt)

    def test__walkdirs_utf8(self):
        tree = ['.bzr', '0file', '1dir/', '1dir/0file', '1dir/1dir/', '2file']
        self.build_tree(tree)
        expected_dirblocks = [(('', '.'), [('0file', '0file', 'file'), ('1dir', '1dir', 'directory'), ('2file', '2file', 'file')]), (('1dir', './1dir'), [('1dir/0file', '0file', 'file'), ('1dir/1dir', '1dir', 'directory')]), (('1dir/1dir', './1dir/1dir'), [])]
        result = []
        found_bzrdir = False
        for dirdetail, dirblock in osutils._walkdirs_utf8(b'.'):
            if len(dirblock) and dirblock[0][1] == b'.bzr':
                found_bzrdir = True
                del dirblock[0]
            dirdetail = (dirdetail[0].decode('utf-8'), osutils.safe_unicode(dirdetail[1]))
            dirblock = [(entry[0].decode('utf-8'), entry[1].decode('utf-8'), entry[2]) for entry in dirblock]
            result.append((dirdetail, dirblock))
        self.assertTrue(found_bzrdir)
        self.assertExpectedBlocks(expected_dirblocks, result)
        result = []
        for dirblock in osutils.walkdirs('./1dir', '1dir'):
            result.append(dirblock)
        self.assertExpectedBlocks(expected_dirblocks[1:], result)

    def _filter_out_stat(self, result):
        """Filter out the stat value from the walkdirs result"""
        for dirdetail, dirblock in result:
            new_dirblock = []
            for info in dirblock:
                new_dirblock.append((info[0], info[1], info[2], info[4]))
            dirblock[:] = new_dirblock

    def _save_platform_info(self):
        self.overrideAttr(osutils, '_selected_dir_reader')

    def assertDirReaderIs(self, expected, top, fs_enc=None):
        """Assert the right implementation for _walkdirs_utf8 is chosen."""
        osutils._selected_dir_reader = None
        self.assertEqual([((b'', top), [])], list(osutils._walkdirs_utf8('.', fs_enc=fs_enc)))
        self.assertIsInstance(osutils._selected_dir_reader, expected)

    def test_force_walkdirs_utf8_fs_utf8(self):
        self.requireFeature(UTF8DirReaderFeature)
        self._save_platform_info()
        self.assertDirReaderIs(UTF8DirReaderFeature.module.UTF8DirReader, b'.', fs_enc='utf-8')

    def test_force_walkdirs_utf8_fs_ascii(self):
        self.requireFeature(UTF8DirReaderFeature)
        self._save_platform_info()
        self.assertDirReaderIs(UTF8DirReaderFeature.module.UTF8DirReader, b'.', fs_enc='ascii')

    def test_force_walkdirs_utf8_fs_latin1(self):
        self._save_platform_info()
        self.assertDirReaderIs(osutils.UnicodeDirReader, '.', fs_enc='iso-8859-1')

    def test_force_walkdirs_utf8_nt(self):
        self.requireFeature(test__walkdirs_win32.win32_readdir_feature)
        self._save_platform_info()
        from .._walkdirs_win32 import Win32ReadDir
        self.assertDirReaderIs(Win32ReadDir, '.')

    def test_unicode_walkdirs(self):
        """Walkdirs should always return unicode paths."""
        self.requireFeature(features.UnicodeFilenameFeature)
        name0 = '0file-¶'
        name1 = '1dir-جو'
        name2 = '2file-س'
        tree = [name0, name1 + '/', name1 + '/' + name0, name1 + '/' + name1 + '/', name2]
        self.build_tree(tree)
        expected_dirblocks = [(('', '.'), [(name0, name0, 'file', './' + name0), (name1, name1, 'directory', './' + name1), (name2, name2, 'file', './' + name2)]), ((name1, './' + name1), [(name1 + '/' + name0, name0, 'file', './' + name1 + '/' + name0), (name1 + '/' + name1, name1, 'directory', './' + name1 + '/' + name1)]), ((name1 + '/' + name1, './' + name1 + '/' + name1), [])]
        result = list(osutils.walkdirs('.'))
        self._filter_out_stat(result)
        self.assertEqual(expected_dirblocks, result)
        result = list(osutils.walkdirs('./' + name1, name1))
        self._filter_out_stat(result)
        self.assertEqual(expected_dirblocks[1:], result)

    def test_unicode__walkdirs_utf8(self):
        """Walkdirs_utf8 should always return utf8 paths.

        The abspath portion might be in unicode or utf-8
        """
        self.requireFeature(features.UnicodeFilenameFeature)
        name0 = '0file-¶'
        name1 = '1dir-جو'
        name2 = '2file-س'
        tree = [name0, name1 + '/', name1 + '/' + name0, name1 + '/' + name1 + '/', name2]
        self.build_tree(tree)
        name0 = name0.encode('utf8')
        name1 = name1.encode('utf8')
        name2 = name2.encode('utf8')
        expected_dirblocks = [((b'', b'.'), [(name0, name0, 'file', b'./' + name0), (name1, name1, 'directory', b'./' + name1), (name2, name2, 'file', b'./' + name2)]), ((name1, b'./' + name1), [(name1 + b'/' + name0, name0, 'file', b'./' + name1 + b'/' + name0), (name1 + b'/' + name1, name1, 'directory', b'./' + name1 + b'/' + name1)]), ((name1 + b'/' + name1, b'./' + name1 + b'/' + name1), [])]
        result = []
        for dirdetail, dirblock in osutils._walkdirs_utf8('.'):
            self.assertIsInstance(dirdetail[0], bytes)
            if isinstance(dirdetail[1], str):
                dirdetail = (dirdetail[0], dirdetail[1].encode('utf8'))
                dirblock = [list(info) for info in dirblock]
                for info in dirblock:
                    self.assertIsInstance(info[4], str)
                    info[4] = info[4].encode('utf8')
            new_dirblock = []
            for info in dirblock:
                self.assertIsInstance(info[0], bytes)
                self.assertIsInstance(info[1], bytes)
                self.assertIsInstance(info[4], bytes)
                new_dirblock.append((info[0], info[1], info[2], info[4]))
            result.append((dirdetail, new_dirblock))
        self.assertEqual(expected_dirblocks, result)

    def test__walkdirs_utf8_with_unicode_fs(self):
        """UnicodeDirReader should be a safe fallback everywhere

        The abspath portion should be in unicode
        """
        self.requireFeature(features.UnicodeFilenameFeature)
        self._save_platform_info()
        osutils._selected_dir_reader = osutils.UnicodeDirReader()
        name0u = '0file-¶'
        name1u = '1dir-جو'
        name2u = '2file-س'
        tree = [name0u, name1u + '/', name1u + '/' + name0u, name1u + '/' + name1u + '/', name2u]
        self.build_tree(tree)
        name0 = name0u.encode('utf8')
        name1 = name1u.encode('utf8')
        name2 = name2u.encode('utf8')
        expected_dirblocks = [((b'', '.'), [(name0, name0, 'file', './' + name0u), (name1, name1, 'directory', './' + name1u), (name2, name2, 'file', './' + name2u)]), ((name1, './' + name1u), [(name1 + b'/' + name0, name0, 'file', './' + name1u + '/' + name0u), (name1 + b'/' + name1, name1, 'directory', './' + name1u + '/' + name1u)]), ((name1 + b'/' + name1, './' + name1u + '/' + name1u), [])]
        result = list(osutils._walkdirs_utf8('.'))
        self._filter_out_stat(result)
        self.assertEqual(expected_dirblocks, result)

    def test__walkdirs_utf8_win32readdir(self):
        self.requireFeature(test__walkdirs_win32.win32_readdir_feature)
        self.requireFeature(features.UnicodeFilenameFeature)
        from .._walkdirs_win32 import Win32ReadDir
        self._save_platform_info()
        osutils._selected_dir_reader = Win32ReadDir()
        name0u = '0file-¶'
        name1u = '1dir-جو'
        name2u = '2file-س'
        tree = [name0u, name1u + '/', name1u + '/' + name0u, name1u + '/' + name1u + '/', name2u]
        self.build_tree(tree)
        name0 = name0u.encode('utf8')
        name1 = name1u.encode('utf8')
        name2 = name2u.encode('utf8')
        expected_dirblocks = [(('', '.'), [(name0, name0, 'file', './' + name0u), (name1, name1, 'directory', './' + name1u), (name2, name2, 'file', './' + name2u)]), ((name1, './' + name1u), [(name1 + '/' + name0, name0, 'file', './' + name1u + '/' + name0u), (name1 + '/' + name1, name1, 'directory', './' + name1u + '/' + name1u)]), ((name1 + '/' + name1, './' + name1u + '/' + name1u), [])]
        result = list(osutils._walkdirs_utf8('.'))
        self._filter_out_stat(result)
        self.assertEqual(expected_dirblocks, result)

    def assertStatIsCorrect(self, path, win32stat):
        os_stat = os.stat(path)
        self.assertEqual(os_stat.st_size, win32stat.st_size)
        self.assertAlmostEqual(os_stat.st_mtime, win32stat.st_mtime, places=4)
        self.assertAlmostEqual(os_stat.st_ctime, win32stat.st_ctime, places=4)
        self.assertAlmostEqual(os_stat.st_atime, win32stat.st_atime, places=4)
        self.assertEqual(os_stat.st_dev, win32stat.st_dev)
        self.assertEqual(os_stat.st_ino, win32stat.st_ino)
        self.assertEqual(os_stat.st_mode, win32stat.st_mode)

    def test__walkdirs_utf_win32_find_file_stat_file(self):
        """make sure our Stat values are valid"""
        self.requireFeature(test__walkdirs_win32.win32_readdir_feature)
        self.requireFeature(features.UnicodeFilenameFeature)
        from .._walkdirs_win32 import Win32ReadDir
        name0u = '0file-¶'
        name0 = name0u.encode('utf8')
        self.build_tree([name0u])
        time.sleep(2)
        with open(name0u, 'ab') as f:
            f.write(b'just a small update')
        result = Win32ReadDir().read_dir('', '.')
        entry = result[0]
        self.assertEqual((name0, name0, 'file'), entry[:3])
        self.assertEqual('./' + name0u, entry[4])
        self.assertStatIsCorrect(entry[4], entry[3])
        self.assertNotEqual(entry[3].st_mtime, entry[3].st_ctime)

    def test__walkdirs_utf_win32_find_file_stat_directory(self):
        """make sure our Stat values are valid"""
        self.requireFeature(test__walkdirs_win32.win32_readdir_feature)
        self.requireFeature(features.UnicodeFilenameFeature)
        from .._walkdirs_win32 import Win32ReadDir
        name0u = '0dir-جو'
        name0 = name0u.encode('utf8')
        self.build_tree([name0u + '/'])
        result = Win32ReadDir().read_dir('', '.')
        entry = result[0]
        self.assertEqual((name0, name0, 'directory'), entry[:3])
        self.assertEqual('./' + name0u, entry[4])
        self.assertStatIsCorrect(entry[4], entry[3])

    def assertPathCompare(self, path_less, path_greater):
        """check that path_less and path_greater compare correctly."""
        self.assertEqual(0, osutils.compare_paths_prefix_order(path_less, path_less))
        self.assertEqual(0, osutils.compare_paths_prefix_order(path_greater, path_greater))
        self.assertEqual(-1, osutils.compare_paths_prefix_order(path_less, path_greater))
        self.assertEqual(1, osutils.compare_paths_prefix_order(path_greater, path_less))

    def test_compare_paths_prefix_order(self):
        self.assertPathCompare('/', '/a')
        self.assertPathCompare('/a', '/b')
        self.assertPathCompare('/b', '/z')
        self.assertPathCompare('/z', '/a/a')
        self.assertPathCompare('/a/b/c', '/d/g')
        self.assertPathCompare('/a/z', '/z/z')
        self.assertPathCompare('/a/c/z', '/a/d/e')
        self.assertPathCompare('', 'a')
        self.assertPathCompare('a', 'b')
        self.assertPathCompare('b', 'z')
        self.assertPathCompare('z', 'a/a')
        self.assertPathCompare('a/b/c', 'd/g')
        self.assertPathCompare('a/z', 'z/z')
        self.assertPathCompare('a/c/z', 'a/d/e')

    def test_path_prefix_sorting(self):
        """Doing a sort on path prefix should match our sample data."""
        original_paths = ['a', 'a/b', 'a/b/c', 'b', 'b/c', 'd', 'd/e', 'd/e/f', 'd/f', 'd/g', 'g']
        dir_sorted_paths = ['a', 'b', 'd', 'g', 'a/b', 'a/b/c', 'b/c', 'd/e', 'd/f', 'd/g', 'd/e/f']
        self.assertEqual(dir_sorted_paths, sorted(original_paths, key=osutils.path_prefix_key))
        self.assertEqual(dir_sorted_paths, sorted(original_paths, key=osutils.path_prefix_key))