import numpy as np
from . import spm99analyze as spm99  # module import
class Spm2AnalyzeImage(spm99.Spm99AnalyzeImage):
    """Class for SPM2 variant of basic Analyze image"""
    header_class = Spm2AnalyzeHeader
    header: Spm2AnalyzeHeader