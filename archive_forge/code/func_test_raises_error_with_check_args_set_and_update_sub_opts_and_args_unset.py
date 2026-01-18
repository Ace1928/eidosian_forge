from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from unittest import mock
from gslib import command
from gslib.tests import testcase
from gslib.utils import constants
def test_raises_error_with_check_args_set_and_update_sub_opts_and_args_unset(self):
    with self.assertRaisesRegex(TypeError, 'Requested to check arguments but sub_opts and args have not been updated.'):
        self._fake_command.ParseSubOpts(check_args=True, should_update_sub_opts_and_args=False)