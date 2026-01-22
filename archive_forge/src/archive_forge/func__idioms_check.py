import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _idioms_check(self, valence, words_and_emoticons, i):
    onezero = f'{words_and_emoticons[i - 1]} {words_and_emoticons[i]}'
    twoonezero = '{} {} {}'.format(words_and_emoticons[i - 2], words_and_emoticons[i - 1], words_and_emoticons[i])
    twoone = f'{words_and_emoticons[i - 2]} {words_and_emoticons[i - 1]}'
    threetwoone = '{} {} {}'.format(words_and_emoticons[i - 3], words_and_emoticons[i - 2], words_and_emoticons[i - 1])
    threetwo = '{} {}'.format(words_and_emoticons[i - 3], words_and_emoticons[i - 2])
    sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]
    for seq in sequences:
        if seq in self.constants.SPECIAL_CASE_IDIOMS:
            valence = self.constants.SPECIAL_CASE_IDIOMS[seq]
            break
    if len(words_and_emoticons) - 1 > i:
        zeroone = f'{words_and_emoticons[i]} {words_and_emoticons[i + 1]}'
        if zeroone in self.constants.SPECIAL_CASE_IDIOMS:
            valence = self.constants.SPECIAL_CASE_IDIOMS[zeroone]
    if len(words_and_emoticons) - 1 > i + 1:
        zeroonetwo = '{} {} {}'.format(words_and_emoticons[i], words_and_emoticons[i + 1], words_and_emoticons[i + 2])
        if zeroonetwo in self.constants.SPECIAL_CASE_IDIOMS:
            valence = self.constants.SPECIAL_CASE_IDIOMS[zeroonetwo]
    if threetwo in self.constants.BOOSTER_DICT or twoone in self.constants.BOOSTER_DICT:
        valence = valence + self.constants.B_DECR
    return valence