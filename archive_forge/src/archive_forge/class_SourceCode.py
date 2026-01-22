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
class SourceCode(str):
    _parsed = None

    def parse(self):
        if self._parsed is None:
            self._parsed = ParsedCode(rinterface.parse(self))
        return self._parsed

    def as_namespace(self, name):
        """ Name for the namespace """
        return SignatureTranslatedAnonymousPackage(self, name)