import warnings
import numpy as np
class FortranFormattingError(TypeError, OSError):
    """Indicates that the file ended mid-record.

    Descends from TypeError for backward compatibility.

    """
    pass