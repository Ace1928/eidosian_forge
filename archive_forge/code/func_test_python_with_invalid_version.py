import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_with_invalid_version(self):
    self.write_file('test.py', 'test file')
    config = testutil.AppInfoFake(runtime='python', entrypoint='run_me_some_python!', runtime_config=dict(python_version='invalid_version'))
    self.assertRaises(testutil.InvalidRuntime, self.generate_config_data, appinfo=config)