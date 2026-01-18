import os
import re
from io import BytesIO, StringIO
import yaml
from .. import registry, tests, version_info_formats
from ..bzr.rio import read_stanzas
from ..version_info_formats.format_custom import (CustomVersionInfoBuilder,
from ..version_info_formats.format_python import PythonVersionInfoBuilder
from ..version_info_formats.format_rio import RioVersionInfoBuilder
from ..version_info_formats.format_yaml import YamlVersionInfoBuilder
from . import TestCaseWithTransport
def test_python_null(self):
    wt = self.make_branch_and_tree('branch')
    sio = StringIO()
    builder = PythonVersionInfoBuilder(wt.branch, working_tree=wt)
    builder.generate(sio)
    val = sio.getvalue()
    self.assertContainsRe(val, "'revision_id': None")
    self.assertContainsRe(val, "'revno': '0'")
    self.assertNotContainsString(val, '\n\n\n\n')