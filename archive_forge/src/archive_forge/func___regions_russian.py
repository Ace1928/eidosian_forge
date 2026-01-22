import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __regions_russian(self, word):
    """
        Return the regions RV and R2 which are used by the Russian stemmer.

        In any word, RV is the region after the first vowel,
        or the end of the word if it contains no vowel.

        R2 is the region after the first non-vowel following
        a vowel in R1, or the end of the word if there is no such non-vowel.

        R1 is the region after the first non-vowel following a vowel,
        or the end of the word if there is no such non-vowel.

        :param word: The Russian word whose regions RV and R2 are determined.
        :type word: str or unicode
        :return: the regions RV and R2 for the respective Russian word.
        :rtype: tuple
        :note: This helper method is invoked by the stem method of the subclass
               RussianStemmer. It is not to be invoked directly!

        """
    r1 = ''
    r2 = ''
    rv = ''
    vowels = ('A', 'U', 'E', 'a', 'e', 'i', 'o', 'u', 'y')
    word = word.replace('i^a', 'A').replace('i^u', 'U').replace('e`', 'E')
    for i in range(1, len(word)):
        if word[i] not in vowels and word[i - 1] in vowels:
            r1 = word[i + 1:]
            break
    for i in range(1, len(r1)):
        if r1[i] not in vowels and r1[i - 1] in vowels:
            r2 = r1[i + 1:]
            break
    for i in range(len(word)):
        if word[i] in vowels:
            rv = word[i + 1:]
            break
    r2 = r2.replace('A', 'i^a').replace('U', 'i^u').replace('E', 'e`')
    rv = rv.replace('A', 'i^a').replace('U', 'i^u').replace('E', 'e`')
    return (rv, r2)