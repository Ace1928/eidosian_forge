import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel4
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def vacancies_at(self, position):
    """
        :return: Number of vacant slots up to, and including, ``position``
        """
    vacancies = 0
    for k in range(1, position + 1):
        if not self._slots[k]:
            vacancies += 1
    return vacancies