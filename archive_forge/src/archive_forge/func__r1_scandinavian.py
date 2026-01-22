import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def _r1_scandinavian(self, word, vowels):
    """
        Return the region R1 that is used by the Scandinavian stemmers.

        R1 is the region after the first non-vowel following a vowel,
        or is the null region at the end of the word if there is no
        such non-vowel. But then R1 is adjusted so that the region
        before it contains at least three letters.

        :param word: The word whose region R1 is determined.
        :type word: str or unicode
        :param vowels: The vowels of the respective language that are
                       used to determine the region R1.
        :type vowels: unicode
        :return: the region R1 for the respective word.
        :rtype: unicode
        :note: This helper method is invoked by the respective stem method of
               the subclasses DanishStemmer, NorwegianStemmer, and
               SwedishStemmer. It is not to be invoked directly!

        """
    r1 = ''
    for i in range(1, len(word)):
        if word[i] not in vowels and word[i - 1] in vowels:
            if 3 > len(word[:i + 1]) > 0:
                r1 = word[3:]
            elif len(word[:i + 1]) >= 3:
                r1 = word[i + 1:]
            else:
                return word
            break
    return r1