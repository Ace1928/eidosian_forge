from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
class LogToCloseToBranchCutError(Exception):
    """
    An exception raised when taking log(-x) for some real number x
    Due to numerical inaccuracies, we cannot know in this case whether to take
    -Pi or Pi as imaginary part.
    """
    pass