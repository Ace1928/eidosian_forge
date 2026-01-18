import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
@unittest.skipIf(not yaml_available, 'Test requires PyYAML')
def test_parseDisplay_userdata_add_block(self):
    self.config.declare('foo', ConfigValue(0, int, None, None))
    self.config.declare('bar', ConfigDict())
    test = _display(self.config, 'userdata')
    sys.stdout.write(test)
    self.assertEqual(yaml_load(test), None)