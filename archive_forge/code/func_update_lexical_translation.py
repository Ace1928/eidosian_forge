import warnings
from collections import defaultdict
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel1
from nltk.translate.ibm_model import Counts
def update_lexical_translation(self, count, s, t):
    self.t_given_s[t][s] += count
    self.any_t_given_s[s] += count