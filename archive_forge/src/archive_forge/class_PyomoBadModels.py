import re
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.scripting.pyomo_main import main
from pyomo.core import (
from pyomo.common.tee import capture_output
from io import StringIO
class PyomoBadModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = check_available_solvers('glpk', 'cplex')

    def pyomo(self, cmd, **kwargs):
        args = ['solve'] + re.split('[ ]+', cmd)
        out = kwargs.get('file', None)
        if out is None:
            out = StringIO()
        with capture_output(out):
            os.chdir(currdir)
            output = main(args)
        if not 'file' in kwargs:
            return output.getvalue()
        return output

    def test_uninstantiated_model_linear(self):
        """Run pyomo with "bad" model file.  Should fail gracefully, with
        a perhaps useful-to-the-user message."""
        if not 'glpk' in solvers:
            self.skipTest('glpk solver is not available')
        return
        base = '%s/test_uninstantiated_model' % currdir
        fout, fbase = (join(base, '_linear.out'), join(base, '.txt'))
        self.pyomo('uninstantiated_model_linear.py', file=fout)
        self.assertTrue(cmp(fout, fbase), msg='Files %s and %s differ' % (fout, fbase))

    def test_uninstantiated_model_quadratic(self):
        """Run pyomo with "bad" model file.  Should fail gracefully, with
        a perhaps useful-to-the-user message."""
        if not 'cplex' in solvers:
            self.skipTest("The 'cplex' executable is not available")
        return
        base = '%s/test_uninstantiated_model' % currdir
        fout, fbase = (join(base, '_quadratic.out'), join(base, '.txt'))
        self.pyomo('uninstantiated_model_quadratic.py --solver=cplex', file=fout)
        self.assertTrue(cmp(fout, fbase), msg='Files %s and %s differ' % (fout, fbase))