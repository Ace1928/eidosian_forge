from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def vectorLength(vector):
    """Calculate the length of the given vector.

    Args:
        vector: A 2D tuple.

    Returns:
        The Euclidean length of the vector.
    """
    x, y = vector
    return math.sqrt(x ** 2 + y ** 2)