import pyomo.environ as pyo
from pyomo.contrib.pynumero.examples.callback.reactor_design import model as m
import logging

This example uses an iteration callback to print the values
of the constraint residuals at each iteration of the CyIpopt
solver
