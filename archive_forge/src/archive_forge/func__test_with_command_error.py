import io
import json
import os
import sys
from unittest import mock
import ddt
from osprofiler.cmd import shell
from osprofiler import exc
from osprofiler.tests import test
def _test_with_command_error(self, cmd, expected_message):
    try:
        self.run_command(cmd)
    except exc.CommandError as actual_error:
        self.assertEqual(str(actual_error), expected_message)
    else:
        raise ValueError("Expected: `osprofiler.exc.CommandError` is raised with message: '%s'." % expected_message)