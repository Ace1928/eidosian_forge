import glob
import shutil
from os.path import join, basename, dirname, isfile
from pyomo.opt.base.solvers import _extract_version
Methods used to create a mock MIP solver used for testing