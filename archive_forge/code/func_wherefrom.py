import os
import typing
import warnings
from types import ModuleType
from warnings import warn
import rpy2.rinterface as rinterface
from . import conversion
from rpy2.robjects.functions import (SignatureTranslatedFunction,
from rpy2.robjects import Environment
from rpy2.robjects.packages_utils import (
import rpy2.robjects.help as rhelp
def wherefrom(symbol: str, startenv: rinterface.SexpEnvironment=rinterface.globalenv):
    """ For a given symbol, return the environment
    this symbol is first found in, starting from 'startenv'.
    """
    env = startenv
    while True:
        if symbol in env:
            break
        env = env.enclos
        if env.rsame(rinterface.emptyenv):
            break
    return conversion.get_conversion().rpy2py(env)