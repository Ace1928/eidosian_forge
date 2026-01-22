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
def _raise_ephemeral_error(name, keyword=''):
    raise AttributeError("The property '%s' can no longer be set directly on the solver object. It should instead be passed as a keyword into the solve method%s. It will automatically be reset to its default value after each invocation of solve." % (name, keyword))