import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
def test_nothing_set(self):
    self.assertEqual(self.expected_params, utils.common_params_for_list(self.args, [], []))