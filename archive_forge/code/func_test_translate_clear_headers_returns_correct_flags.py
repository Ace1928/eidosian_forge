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
@mock.patch.object(shim_util, 'COMMANDS_SUPPORTING_ALL_HEADERS', new={'fake_shim'})
def test_translate_clear_headers_returns_correct_flags(self):
    flags = self._fake_command._translate_headers({'Cache-Control': 'fake_Cache_Control'}, unset=True)
    self.assertCountEqual(flags, ['--clear-cache-control'])