import math
import sys
import copy
import json
import logging
import os.path
from pyomo.common.dependencies import yaml, yaml_load_args
import pyomo.opt
from pyomo.opt.results.container import undefined, ignore, ListContainer, MapContainer
import pyomo.opt.results.solution
from pyomo.opt.results.solution import default_print_options as dpo
import pyomo.opt.results.problem
import pyomo.opt.results.solver
from io import StringIO
def _repn_(self, option):
    if not option.schema and (not self._active) and (not self._required):
        return ignore
    tmp = {}
    for key in self._sections:
        rep = dict.__getitem__(self, key)._repn_(option)
        if not rep == ignore:
            tmp[key] = rep
    return tmp