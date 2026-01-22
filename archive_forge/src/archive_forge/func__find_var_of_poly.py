import re
from .utilities import MethodMappingList
from .component import NonZeroDimensionalComponent
from .coordinates import PtolemyCoordinates
from .rur import RUR
from . import processFileBase
from ..pari import pari
def _find_var_of_poly(text):
    return re.search('[_A-Za-z][_A-Za-z0-9]*', text).group(0)