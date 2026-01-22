from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class TypFitArguments:
    """Table 8.2 of the PDF 1.7 reference."""
    FIT = '/Fit'
    FIT_V = '/FitV'
    FIT_BV = '/FitBV'
    FIT_B = '/FitB'
    FIT_H = '/FitH'
    FIT_BH = '/FitBH'
    FIT_R = '/FitR'
    XYZ = '/XYZ'