from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
def _getCurrentPoint(self):
    """Return the current point. This is not part of the public
        interface, yet is useful for subclasses.
        """
    return self.__currentPoint