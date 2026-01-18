from pyomo.common.dependencies import numpy as np, pandas as pd
from itertools import product
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.semibatch.semibatch import generate_model

The following script can be used to run semibatch parameter estimation in 
parallel and save results to files for later analysis and graphics.
Example command: mpiexec -n 4 python parallel_example.py
