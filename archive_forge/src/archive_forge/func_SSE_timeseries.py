from pyomo.common.dependencies import pandas as pd
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
def SSE_timeseries(model, data):
    expr = 0
    for val in data['ca']:
        expr = expr + (float(val) - model.ca) ** 2 * (1 / len(data['ca']))
    for val in data['cb']:
        expr = expr + (float(val) - model.cb) ** 2 * (1 / len(data['cb']))
    for val in data['cc']:
        expr = expr + (float(val) - model.cc) ** 2 * (1 / len(data['cc']))
    for val in data['cd']:
        expr = expr + (float(val) - model.cd) ** 2 * (1 / len(data['cd']))
    return expr