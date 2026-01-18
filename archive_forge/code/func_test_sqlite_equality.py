import json
import os
import pyomo.common.unittest as unittest
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers
@unittest.skipUnless(sqlite3_available, 'Requires SQLite3')
def test_sqlite_equality(self):
    dat_results_file = self.run_pyomo(os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet.dat'), outputpath=os.path.join(currdir, 'dat_results.jsn'))
    with open(dat_results_file) as FILE:
        dat_results = json.load(FILE)
    sqlite_results_file = self.run_pyomo(os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet1.sqlite.dat'), outputpath=os.path.join(currdir, 'sqlite_results.jsn'))
    with open(sqlite_results_file) as FILE:
        sqlite_results = json.load(FILE)
    del dat_results['Solver'][0]['Time']
    del sqlite_results['Solver'][0]['Time']
    self.assertStructuredAlmostEqual(dat_results, sqlite_results)
    os.remove(dat_results_file)
    os.remove(sqlite_results_file)