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
class SignatureTranslatedAnonymousPackage(SignatureTranslatedPackage):

    def __init__(self, string, name):
        env = Environment()
        reval(string, env)
        super(SignatureTranslatedAnonymousPackage, self).__init__(env, name)