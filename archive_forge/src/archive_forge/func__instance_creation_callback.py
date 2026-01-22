import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def _instance_creation_callback(self, experiment_number=None, cb_data=None):
    exp_data = cb_data[experiment_number]
    if isinstance(exp_data, (dict, pd.DataFrame)):
        pass
    elif isinstance(exp_data, str):
        try:
            with open(exp_data, 'r') as infile:
                exp_data = json.load(infile)
        except:
            raise RuntimeError(f'Could not read {exp_data} as json')
    else:
        raise RuntimeError(f'Unexpected data format for cb_data={cb_data}')
    model = self._create_parmest_model(exp_data)
    return model