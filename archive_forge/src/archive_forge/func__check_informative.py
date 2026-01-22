import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI
def _check_informative(self, feat, flag=False):
    """
        Check whether a feature is informative
        The flag control whether "_" is informative or not
        """
    if feat is None:
        return False
    if feat == '':
        return False
    if flag is False:
        if feat == '_':
            return False
    return True