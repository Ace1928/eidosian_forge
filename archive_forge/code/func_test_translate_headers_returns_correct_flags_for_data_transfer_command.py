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
def test_translate_headers_returns_correct_flags_for_data_transfer_command(self):
    self._fake_command.headers = {'Cache-Control': 'fake_Cache_Control', 'Content-Disposition': 'fake_Content-Disposition', 'Content-Encoding': 'fake_Content-Encoding', 'Content-Language': 'fake_Content-Language', 'Content-Type': 'fake_Content-Type', 'Content-MD5': 'fake_Content-MD5', 'custom-time': 'fake_time', 'x-goog-if-generation-match': 'fake_gen_match', 'x-goog-if-metageneration-match': 'fake_metagen_match', 'x-goog-meta-cAsE': 'sEnSeTiVe', 'x-goog-meta-gfoo': 'fake_goog_meta', 'x-amz-meta-afoo': 'fake_amz_meta', 'x-amz-afoo': 'fake_amz_custom_header'}
    flags = self._fake_command._translate_headers()
    self.assertCountEqual(flags, ['--cache-control=fake_Cache_Control', '--content-disposition=fake_Content-Disposition', '--content-encoding=fake_Content-Encoding', '--content-language=fake_Content-Language', '--content-type=fake_Content-Type', '--content-md5=fake_Content-MD5', '--custom-time=fake_time', '--if-generation-match=fake_gen_match', '--if-metageneration-match=fake_metagen_match', '--update-custom-metadata=cAsE=sEnSeTiVe', '--update-custom-metadata=gfoo=fake_goog_meta', '--update-custom-metadata=afoo=fake_amz_meta', '--additional-headers=x-amz-afoo=fake_amz_custom_header'])