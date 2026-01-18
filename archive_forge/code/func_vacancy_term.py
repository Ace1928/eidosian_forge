import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel4
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def vacancy_term(i):
    value = 1.0
    tablet = alignment_info.cepts[i]
    tablet_length = len(tablet)
    total_vacancies = slots.vacancies_at(len(slots))
    if tablet_length == 0:
        return value
    j = tablet[0]
    previous_cept = alignment_info.previous_cept(j)
    previous_center = alignment_info.center_of_cept(previous_cept)
    dv = slots.vacancies_at(j) - slots.vacancies_at(previous_center)
    max_v = total_vacancies - tablet_length + 1
    trg_class = self.trg_classes[alignment_info.trg_sentence[j]]
    value *= self.head_vacancy_table[dv][max_v][trg_class]
    slots.occupy(j)
    total_vacancies -= 1
    if value < MIN_PROB:
        return MIN_PROB
    for k in range(1, tablet_length):
        previous_position = tablet[k - 1]
        previous_vacancies = slots.vacancies_at(previous_position)
        j = tablet[k]
        dv = slots.vacancies_at(j) - previous_vacancies
        max_v = total_vacancies - tablet_length + k + 1 - previous_vacancies
        trg_class = self.trg_classes[alignment_info.trg_sentence[j]]
        value *= self.non_head_vacancy_table[dv][max_v][trg_class]
        slots.occupy(j)
        total_vacancies -= 1
        if value < MIN_PROB:
            return MIN_PROB
    return value