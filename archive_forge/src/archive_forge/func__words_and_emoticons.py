import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _words_and_emoticons(self):
    """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
    wes = self.text.split()
    words_punc_dict = self._words_plus_punc()
    wes = [we for we in wes if len(we) > 1]
    for i, we in enumerate(wes):
        if we in words_punc_dict:
            wes[i] = words_punc_dict[we]
    return wes