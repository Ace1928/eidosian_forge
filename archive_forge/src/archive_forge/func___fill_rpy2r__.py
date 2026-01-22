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
def __fill_rpy2r__(self, on_conflict='fail'):
    super(SignatureTranslatedPackage, self).__fill_rpy2r__(on_conflict=on_conflict)
    for name, robj in self.__dict__.items():
        if isinstance(robj, rinterface.Sexp) and robj.typeof == rinterface.RTYPES.CLOSXP:
            self.__dict__[name] = DocumentedSTFunction(self.__dict__[name], packagename=self.__rname__)