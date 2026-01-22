import re
from unittest import mock
from testtools import matchers
from magnumclient.tests import utils
def _test_arg_success(self, command, keyword=None):
    stdout, stderr = self.shell(command)
    if keyword:
        self.assertIn(keyword, stdout + stderr)