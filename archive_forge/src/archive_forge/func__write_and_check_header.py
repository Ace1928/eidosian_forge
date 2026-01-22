import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.environ as pe
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
import os
def _write_and_check_header(self, m, correct_lines):
    writer = appsi.writers.NLWriter()
    with TempfileManager:
        fname = TempfileManager.create_tempfile(suffix='.appsi.nl')
        writer.write(m, fname)
        with open(fname, 'r') as f:
            for ndx, line in enumerate(list(f.readlines())[:10]):
                self.assertTrue(line.startswith(correct_lines[ndx]))