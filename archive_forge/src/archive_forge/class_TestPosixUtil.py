from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.tests import testcase
from gslib.tests.util import unittest
from gslib.utils import posix_util
from gslib.utils.system_util import IS_WINDOWS
from six import add_move, MovedModule
from six.moves import mock
class TestPosixUtil(testcase.GsUtilUnitTestCase):
    """Unit tests for POSIX utils."""

    @mock.patch.object(posix_util, 'InitializeUserGroups', autospec=True)
    @mock.patch.object(posix_util, 'InitializeDefaultMode', autospec=True)
    def test_initialize_preserve_posix_data_calls_correct_functions(self, mock_initialize_default_mode, mock_initialize_user_groups):
        posix_util.InitializePreservePosixData()
        mock_initialize_default_mode.assert_called_once_with()
        mock_initialize_user_groups.assert_called_once_with()

    @unittest.skipIf(IS_WINDOWS, 'os.umask always returns 0 on Windows.')
    @mock.patch.object(os, 'umask', autospec=True)
    def test_initialize_mode_sets_umask_to_correct_temporary_value_not_windows(self, mock_umask):
        mock_umask.side_effect = ValueError
        with self.assertRaises(ValueError):
            posix_util.InitializeDefaultMode()
        mock_umask.assert_called_once_with(127)