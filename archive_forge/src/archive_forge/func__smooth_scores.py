import math
import re
from nltk.tokenize.api import TokenizerI
def _smooth_scores(self, gap_scores):
    """Wraps the smooth function from the SciPy Cookbook"""
    return list(smooth(numpy.array(gap_scores[:]), window_len=self.smoothing_width + 1))