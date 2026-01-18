from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from contextlib import contextmanager
import os
import re
import subprocess
from unittest import mock
from boto import config
from gslib import command
from gslib import command_argument
from gslib import exception
from gslib.commands import rsync
from gslib.commands import version
from gslib.commands import test
from gslib.cs_api_map import ApiSelector
from gslib.tests import testcase
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.tests import util
@mock.patch.object(shim_util, 'PRECONDITONS_ONLY_SUPPORTED_COMMANDS', new={'fake_shim'})
def test_translate_valid_headers_for_precondition_supported_command(self):
    self._fake_command.headers = {'x-goog-if-generation-match': 'fake_gen_match', 'x-goog-if-metageneration-match': 'fake_metagen_match', 'x-goog-meta-foo': 'fake_goog_meta'}
    flags = self._fake_command._translate_headers()
    self.assertCountEqual(flags, ['--if-generation-match=fake_gen_match', '--if-metageneration-match=fake_metagen_match'])