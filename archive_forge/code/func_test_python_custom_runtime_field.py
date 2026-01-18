import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_custom_runtime_field(self):
    """Verify that a runtime field of "custom" works."""
    self.write_file('test.py', 'test file')
    config = testutil.AppInfoFake(runtime='custom', entrypoint='my_entrypoint')
    self.assertTrue(self.generate_configs(appinfo=config, deploy=True))