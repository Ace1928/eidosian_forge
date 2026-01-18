from . import matrix
from . import homology
from .polynomial import Polynomial
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase, processFileDispatch, processMagmaFile
from . import utilities
from string import Template
import signal
import re
import os
import sys
from urllib.request import Request, urlopen
from urllib.request import quote as urlquote
from urllib.error import HTTPError
def to_magma(self, template_path='magma/default.magma_template'):
    """
        Returns a string with the ideal that can be used as input for magma.

        The advanced user can provide a template string to write own magma
        code to process the equations.

        >>> from snappy import *
        >>> p = Manifold("4_1").ptolemy_variety(2, obstruction_class = 1)

        Magma input to compute radical Decomposition
        >>> s = p.to_magma()
        >>> print(s.split('ring and ideal')[1].strip())    #doctest: +ELLIPSIS  +NORMALIZE_WHITESPACE
        R<c_0011_0, c_0101_0> := PolynomialRing(RationalField(), 2, "grevlex");
        MyIdeal := ideal<R | - c_0011_0 * c_0101_0 + c_0011_0^2 + c_0101_0^2,
        ...
        >>> "RadicalDecomposition" in p.to_magma()
        True
        """
    if os.path.isfile(template_path):
        template = open(template_path, 'r').read()
    else:
        from snappy.ptolemy import __path__ as base_paths
        abs_path = os.path.join(base_paths[0], template_path)
        if os.path.isfile(abs_path):
            template = open(abs_path, 'r').read()
        else:
            raise Exception('No file at template_path %s' % template_path)
    PREAMBLE = '==TRIANGULATION=BEGINS==\n' + self._manifold._to_string() + '\n==TRIANGULATION=ENDS==\n' + 'PY=EVAL=SECTION=BEGINS=HERE\n' + self.py_eval_section() + '\nPY=EVAL=SECTION=ENDS=HERE\n'
    QUOTED_PREAMBLE = utilities.quote_ascii_text(utilities.break_long_lines(PREAMBLE))
    return Template(template).safe_substitute(PREAMBLE=PREAMBLE, QUOTED_PREAMBLE=QUOTED_PREAMBLE, VARIABLES=', '.join(self.variables), VARIABLES_QUOTED=', '.join(['"%s"' % v for v in self.variables]), VARIABLE_NUMBER=len(self.variables), VARIABLES_WITH_NON_ZERO_CONDITION=', '.join(self.variables_with_non_zero_condition), VARIABLES_WITH_NON_ZERO_CONDITION_QUOTED=', '.join(['"%s"' % v for v in self.variables_with_non_zero_condition]), VARIABLE_WITH_NON_ZERO_CONDITION_NUMBER=len(self.variables_with_non_zero_condition), EQUATIONS=',\n          '.join([str(eqn) for eqn in self.equations]), EQUATIONS_WITH_NON_ZERO_CONDITION=',\n          '.join([str(eqn) for eqn in self.equations_with_non_zero_condition]))