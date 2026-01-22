import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
class Model4Counts(Counts):
    """
    Data object to store counts of various parameters during training.
    Includes counts for distortion.
    """

    def __init__(self):
        super().__init__()
        self.head_distortion = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        self.head_distortion_for_any_dj = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.non_head_distortion = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.non_head_distortion_for_any_dj = defaultdict(lambda: 0.0)

    def update_distortion(self, count, alignment_info, j, src_classes, trg_classes):
        i = alignment_info.alignment[j]
        t = alignment_info.trg_sentence[j]
        if i == 0:
            pass
        elif alignment_info.is_head_word(j):
            previous_cept = alignment_info.previous_cept(j)
            if previous_cept is not None:
                previous_src_word = alignment_info.src_sentence[previous_cept]
                src_class = src_classes[previous_src_word]
            else:
                src_class = None
            trg_class = trg_classes[t]
            dj = j - alignment_info.center_of_cept(previous_cept)
            self.head_distortion[dj][src_class][trg_class] += count
            self.head_distortion_for_any_dj[src_class][trg_class] += count
        else:
            previous_j = alignment_info.previous_in_tablet(j)
            trg_class = trg_classes[t]
            dj = j - previous_j
            self.non_head_distortion[dj][trg_class] += count
            self.non_head_distortion_for_any_dj[trg_class] += count