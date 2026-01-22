import numpy as np
class IncompatibleCellError(ValueError):
    """Exception raised if stacking fails due to incompatible cells
    between *atoms1* and *atoms2*."""
    pass