import json
import pkgutil
import operator
from typing import List
from deprecated import deprecated
from deprecated.sphinx import versionadded
from lark import Lark, Transformer, v_args
import numpy as np
from pyquil.quilbase import (
from pyquil.quiltwaveforms import _wf_from_dict
from pyquil.quilatom import (
from pyquil.gates import (
@v_args(inline=True)
def apply_fun(self, fun, arg):
    if fun.upper() == 'SIN':
        return quil_sin(arg) if isinstance(arg, Expression) else np.sin(arg)
    if fun.upper() == 'COS':
        return quil_cos(arg) if isinstance(arg, Expression) else np.cos(arg)
    if fun.upper() == 'SQRT':
        return quil_sqrt(arg) if isinstance(arg, Expression) else np.sqrt(arg)
    if fun.upper() == 'EXP':
        return quil_exp(arg) if isinstance(arg, Expression) else np.exp(arg)
    if fun.upper() == 'CIS':
        return quil_cis(arg) if isinstance(arg, Expression) else np.cos(arg) + 1j * np.sin(arg)