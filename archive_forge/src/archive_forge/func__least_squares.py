import sys
import pyomo.environ as pyo
import numpy.random as rnd
from pyomo.common.dependencies import pandas as pd
import pyomo.contrib.pynumero.examples.external_grey_box.param_est.models as po
def _least_squares(m):
    obj = 0
    for i in m.PTS:
        row = m.df.iloc[i]
        obj += (m.model_i[i].egb.inputs['Th_in'] - float(row['Th_in'])) ** 2
        obj += (m.model_i[i].egb.inputs['Tc_in'] - float(row['Tc_in'])) ** 2
        obj += (m.model_i[i].egb.inputs['Th_out'] - float(row['Th_out'])) ** 2
        obj += (m.model_i[i].egb.inputs['Tc_out'] - float(row['Tc_out'])) ** 2
    return obj