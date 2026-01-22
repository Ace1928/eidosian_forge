from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _get_obstruction_variable(self, face, tet):
    """
        Get the obstruction variable sigma_face for the given face and
        tetrahedron. Return 1 if there is no such obstruction class.
        """
    key = 's_%d_%d' % (face, tet)
    return self.get(key, +1)