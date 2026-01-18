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
def test_unusedUserValues_scalar(self):
    self.config['scenario']['merlion'] = True
    test = '\n'.join((x.name(True) for x in self.config.unused_user_values()))
    sys.stdout.write(test)
    self.assertEqual(test, 'scenario.merlion')