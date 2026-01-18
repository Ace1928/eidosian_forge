from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from unittest import mock
from gslib import command
from gslib.tests import testcase
from gslib.utils import constants
def test_uses_self_args_if_args_passed_is_None(self):
    args_list = ['fake', 'args']
    self._fake_command.args = args_list
    _, parsed_args = self._fake_command.ParseSubOpts(should_update_sub_opts_and_args=False)
    self.assertEqual(parsed_args, args_list)