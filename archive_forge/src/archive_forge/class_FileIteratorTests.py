from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import six
import tempfile
from gslib import wildcard_iterator
from gslib.exception import InvalidUrlError
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetDummyProjectForUnitTest
class FileIteratorTests(testcase.GsUtilUnitTestCase):
    """Unit tests for FileWildcardIterator."""

    def setUp(self):
        """Creates a test dir with 3 files and one nested subdirectory + file."""
        super(FileIteratorTests, self).setUp()
        self.test_dir = self.CreateTempDir(test_files=['abcd', 'abdd', 'ade$', ('dir1', 'dir2', 'zzz')])
        self.root_files_uri_strs = set([suri(self.test_dir, 'abcd'), suri(self.test_dir, 'abdd'), suri(self.test_dir, 'ade$')])
        self.subdirs_uri_strs = set([suri(self.test_dir, 'dir1')])
        self.nested_files_uri_strs = set([suri(self.test_dir, 'dir1', 'dir2', 'zzz')])
        self.immed_child_uri_strs = self.root_files_uri_strs | self.subdirs_uri_strs
        self.all_file_uri_strs = self.root_files_uri_strs | self.nested_files_uri_strs

    def testContainsWildcard(self):
        """Tests ContainsWildcard call."""
        self.assertTrue(ContainsWildcard('a*.txt'))
        self.assertTrue(ContainsWildcard('a[0-9].txt'))
        self.assertFalse(ContainsWildcard('0-9.txt'))
        self.assertTrue(ContainsWildcard('?.txt'))

    def testNoOpDirectoryIterator(self):
        """Tests that directory-only URI iterates just that one URI."""
        results = list(self._test_wildcard_iterator(suri(tempfile.tempdir)).IterAll(expand_top_level_buckets=True))
        self.assertEqual(1, len(results))
        self.assertEqual(suri(tempfile.tempdir), str(results[0]))

    def testMatchingAllFiles(self):
        """Tests matching all files, based on wildcard."""
        uri = self._test_storage_uri(suri(self.test_dir, '*'))
        actual_uri_strs = set((str(u) for u in self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(self.immed_child_uri_strs, actual_uri_strs)

    def testMatchingAllFilesWithSize(self):
        """Tests matching all files, based on wildcard."""
        uri = self._test_storage_uri(suri(self.test_dir, '*'))
        blrs = self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True, bucket_listing_fields=['size'])
        num_expected_objects = 3
        num_actual_objects = 0
        for blr in blrs:
            self.assertTrue(str(blr) in self.immed_child_uri_strs)
            if blr.IsObject():
                num_actual_objects += 1
                self.assertEqual(blr.root_object.size, 6)
        self.assertEqual(num_expected_objects, num_actual_objects)

    def testMatchingFileSubset(self):
        """Tests matching a subset of files, based on wildcard."""
        exp_uri_strs = set([suri(self.test_dir, 'abcd'), suri(self.test_dir, 'abdd')])
        uri = self._test_storage_uri(suri(self.test_dir, 'ab??'))
        actual_uri_strs = set((str(u) for u in self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_uri_strs, actual_uri_strs)

    def testMatchingNonWildcardedUri(self):
        """Tests matching a single named file."""
        exp_uri_strs = set([suri(self.test_dir, 'abcd')])
        uri = self._test_storage_uri(suri(self.test_dir, 'abcd'))
        actual_uri_strs = set((str(u) for u in self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_uri_strs, actual_uri_strs)

    def testMatchingFilesIgnoringOtherRegexChars(self):
        """Tests ignoring non-wildcard regex chars (e.g., ^ and $)."""
        exp_uri_strs = set([suri(self.test_dir, 'ade$')])
        uri = self._test_storage_uri(suri(self.test_dir, 'ad*$'))
        actual_uri_strs = set((str(u) for u in self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_uri_strs, actual_uri_strs)

    def testRecursiveDirectoryOnlyWildcarding(self):
        """Tests recursive expansion of directory-only '**' wildcard."""
        uri = self._test_storage_uri(suri(self.test_dir, '**'))
        actual_uri_strs = set((str(u) for u in self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(self.all_file_uri_strs, actual_uri_strs)

    def testRecursiveDirectoryPlusFileWildcarding(self):
        """Tests recursive expansion of '**' directory plus '*' wildcard."""
        uri = self._test_storage_uri(suri(self.test_dir, '**', '*'))
        actual_uri_strs = set((str(u) for u in self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(self.all_file_uri_strs, actual_uri_strs)

    def testInvalidRecursiveDirectoryWildcard(self):
        """Tests that wildcard containing '***' raises exception."""
        try:
            uri = self._test_storage_uri(suri(self.test_dir, '***', 'abcd'))
            for unused_ in self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True):
                self.fail('Expected WildcardException not raised.')
        except wildcard_iterator.WildcardException as e:
            self.assertTrue(str(e).find('more than 2 consecutive') != -1)

    def testMissingDir(self):
        """Tests that wildcard gets empty iterator when directory doesn't exist."""
        res = list(self._test_wildcard_iterator(suri('no_such_dir', '*')).IterAll(expand_top_level_buckets=True))
        self.assertEqual(0, len(res))

    def testExistingDirNoFileMatch(self):
        """Tests that wildcard returns empty iterator when there's no match."""
        uri = self._test_storage_uri(suri(self.test_dir, 'non_existent*'))
        res = list(self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True))
        self.assertEqual(0, len(res))

    def testExcludeDir(self):
        """Tests that the exclude regex will omit a nested directory."""
        exp_uri_strs = self.root_files_uri_strs
        uri = self._test_storage_uri(suri(self.test_dir, '**'))
        exclude_tuple = (StorageUrlFromString(self.test_dir), True, re.compile('dir1'))
        actual_uri_strs = set((str(u) for u in self._test_wildcard_iterator(uri, exclude_tuple=exclude_tuple).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_uri_strs, actual_uri_strs)

    def testExcludeTupleButExcludeDirFalse(self):
        """Tests that the exclude regex will be disabled by exlude_dirs False."""
        exp_uri_strs = self.all_file_uri_strs
        uri = self._test_storage_uri(suri(self.test_dir, '**'))
        exclude_tuple = (StorageUrlFromString(self.test_dir), False, re.compile('dir1'))
        actual_uri_strs = set((str(u) for u in self._test_wildcard_iterator(uri, exclude_tuple=exclude_tuple).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_uri_strs, actual_uri_strs)