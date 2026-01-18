import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar

Semibatch model, based on Nicholson et al. (2018). pyomo.dae: A modeling and 
automatic discretization framework for optimization with di
erential and 
algebraic equations. Mathematical Programming Computation, 10(2), 187-223.
