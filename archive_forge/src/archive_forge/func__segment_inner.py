import re
from typing import Tuple
from nltk.stem.api import StemmerI
def _segment_inner(self, word: str, upper: bool):
    """Inner method for iteratively applying the code stemming regexes.
        This method receives a pre-processed variant of the word to be stemmed,
        or the word to be segmented, and returns a tuple of the word and the
        removed suffix.

        :param word: A pre-processed variant of the word that is to be stemmed.
        :type word: str
        :param upper: Whether the original word started with a capital letter.
        :type upper: bool
        :return: A tuple of the stemmed word and the removed suffix.
        :rtype: Tuple[str, str]
        """
    rest_length = 0
    word_copy = word[:]
    word = Cistem.replace_to(word)
    rest = ''
    while len(word) > 3:
        if len(word) > 5:
            word, n = Cistem.strip_emr.subn('', word)
            if n != 0:
                rest_length += 2
                continue
            word, n = Cistem.strip_nd.subn('', word)
            if n != 0:
                rest_length += 2
                continue
        if not upper or self._case_insensitive:
            word, n = Cistem.strip_t.subn('', word)
            if n != 0:
                rest_length += 1
                continue
        word, n = Cistem.strip_esn.subn('', word)
        if n != 0:
            rest_length += 1
            continue
        else:
            break
    word = Cistem.replace_back(word)
    if rest_length:
        rest = word_copy[-rest_length:]
    return (word, rest)