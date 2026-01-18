from pyomo.environ import (
import pyomo.contrib.parmest.parmest as parmest
 
Example from Y. Bard, "Nonlinear Parameter Estimation", (pg. 124)

This example shows:
1. How to define the unknown (to be regressed parameters) with an index
2. How to call parmest to only estimate some of the parameters (and fix the rest)

Code provided by Paul Akula.
