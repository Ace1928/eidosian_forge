from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
def _create_block(self):
    """
        Create a pyomo Concrete model and add blocks with different parameter perturbation scenarios.

        Returns
        -------
        mod: Concrete Pyomo model
        """
    scena_gen = ScenarioGenerator(parameter_dict=self.param, formula=self.formula, step=self.step)
    self.scenario_data = scena_gen.ScenarioData
    self.scenario_list = self.scenario_data.scenario
    self.scenario_num = self.scenario_data.scena_num
    self.eps_abs = self.scenario_data.eps_abs
    self.scena_gen = scena_gen
    mod = pyo.ConcreteModel()
    mod.scenario = pyo.Set(initialize=self.scenario_data.scenario_indices)
    self.create_model(mod=mod, model_option=ModelOptionLib.stage1)

    def block_build(b, s):
        self.create_model(mod=b, model_option=ModelOptionLib.stage2)
        for par in self.param:
            cuid = pyo.ComponentUID(par)
            var = cuid.find_component_on(b)
            var.fix(self.scenario_data.scenario[s][par])
    mod.block = pyo.Block(mod.scenario, rule=block_build)
    if self.discretize_model:
        mod = self.discretize_model(mod)
    for name in self.design_name:

        def fix1(mod, s):
            cuid = pyo.ComponentUID(name)
            design_var_global = cuid.find_component_on(mod)
            design_var = cuid.find_component_on(mod.block[s])
            return design_var == design_var_global
        con_name = 'con' + name
        mod.add_component(con_name, pyo.Constraint(mod.scenario, expr=fix1))
    return mod