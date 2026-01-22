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
def _sequential_finite(self, read_output, extract_single_model, store_output):
    """Sequential_finite mode uses Pyomo Block to evaluate the sensitivity information."""
    if read_output:
        with open(read_output, 'rb') as f:
            output_record = pickle.load(f)
            f.close()
        jac = self._finite_calculation(output_record)
    else:
        mod = self._create_block()
        output_record = {}
        square_result = self._solve_doe(mod, fix=True)
        if extract_single_model:
            mod_name = store_output + '.csv'
            dataframe = extract_single_model(mod, square_result)
            dataframe.to_csv(mod_name)
        for s in range(len(self.scenario_list)):
            output_iter = []
            for r in self.measure_name:
                cuid = pyo.ComponentUID(r)
                try:
                    var_up = cuid.find_component_on(mod.block[s])
                except:
                    raise ValueError(f'measurement {r} cannot be found in the model.')
                output_iter.append(pyo.value(var_up))
            output_record[s] = output_iter
            output_record['design'] = self.design_values
            if store_output:
                f = open(store_output, 'wb')
                pickle.dump(output_record, f)
                f.close()
        jac = self._finite_calculation(output_record)
        self.model = mod
        self.jac = jac
    if self.specified_prior is None:
        prior_in_use = self.prior_FIM
    else:
        prior_in_use = self.specified_prior
    FIM_analysis = FisherResults(list(self.param.keys()), self.measurement_vars, jacobian_info=None, all_jacobian_info=jac, prior_FIM=prior_in_use, store_FIM=self.FIM_store_name, scale_constant_value=self.scale_constant_value)
    return FIM_analysis