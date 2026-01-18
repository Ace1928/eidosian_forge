from pyomo.common.dependencies import pandas as pd, pandas_available
import pyomo.common.unittest as unittest
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.scenariocreator as sc
import pyomo.environ as pyo
from pyomo.environ import SolverFactory
def test_semibatch_bootstrap(self):
    scenmaker = sc.ScenarioCreator(self.pest, 'ipopt')
    bootscens = sc.ScenarioSet('Bootstrap')
    numtomake = 2
    scenmaker.ScenariosFromBootstrap(bootscens, numtomake, seed=1134)
    tval = bootscens.ScenarioNumber(0).ThetaVals['k1']
    self.assertAlmostEqual(tval, 20.64, places=1)