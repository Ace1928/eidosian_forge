import warnings
from collections import defaultdict
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel1
from nltk.translate.ibm_model import Counts
def update_alignment(self, count, i, j, l, m):
    self.alignment[i][j][l][m] += count
    self.alignment_for_any_i[j][l][m] += count