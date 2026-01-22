from __future__ import annotations
from typing import Callable, Optional
from collections import OrderedDict
import os
import re
import subprocess
from .util import (
class FortranCompilerRunner(CompilerRunner):
    standards = (None, 'f77', 'f95', 'f2003', 'f2008')
    std_formater = {'gfortran': lambda x: '-std=gnu' if x is None else '-std=legacy' if x == 'f77' else '-std={}'.format(x), 'ifort': lambda x: '-stand f08' if x is None else '-stand f{}'.format(x[-2:])}
    compiler_dict = OrderedDict([('gnu', 'gfortran'), ('intel', 'ifort')])
    compiler_name_vendor_mapping = {'gfortran': 'gnu', 'ifort': 'intel'}