import re
import sys
import time
import logging
import shlex
from pyomo.common import Factory
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.opt.base.convert import convert_problem
from pyomo.opt.base.formats import ResultsFormat
import pyomo.opt.base.results
@warm_start_file_name.setter
def warm_start_file_name(self, val):
    _raise_ephemeral_error('warm_start_file_name', keyword=' (warmstart_file)')