from itertools import zip_longest
import json
import re
import os
import sys
from os.path import abspath, dirname, join
from filecmp import cmp
import subprocess
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.core
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers
from io import StringIO
@unittest.skipIf(not yaml_available, 'YAML not available available')
class TestWithYaml(BaseTester):

    def compare_json(self, file1, file2):
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            f1_contents = json.load(f1)
            f2_contents = json.load(f2)
            self.assertStructuredAlmostEqual(f2_contents, f1_contents, abstol=_diff_tol, allow_second_superset=True)

    def test15b_simple_pyomo_execution(self):
        self.pyomo(join(currdir, 'test15b.yaml'), root=join(currdir, 'test15b'))
        self.compare_json(join(currdir, 'test15b.jsn'), join(currdir, 'test1.txt'))

    def test15c_simple_pyomo_execution(self):
        self.pyomo(join(currdir, 'test15c.yaml'), root=join(currdir, 'test15c'))
        self.compare_json(join(currdir, 'test15c.jsn'), join(currdir, 'test1.txt'))