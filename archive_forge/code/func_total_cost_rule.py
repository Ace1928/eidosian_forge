import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost