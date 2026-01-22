import os
import sys
import time
from pyomo.common.dependencies import mpi4py
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo

To run this example:

mpirun -np 3 python farmer.py
