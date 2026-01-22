import os
import shutil
import subprocess
import tempfile
from ..printing import latex
from ..kinetics.rates import RadiolyticBase
from ..units import to_unitless, get_derived_unit
def _doi(s):
    return '\\texttt{\\href{http://dx.doi.org/' + s + '}{doi:' + s + '}}'