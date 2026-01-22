import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
class NorwegianStemmer(_ScandinavianStemmer):
    """
    The Norwegian Snowball stemmer.

    :cvar __vowels: The Norwegian vowels.
    :type __vowels: unicode
    :cvar __s_ending: Letters that may directly appear before a word final 's'.
    :type __s_ending: unicode
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :note: A detailed description of the Norwegian
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/norwegian/stemmer.html

    """
    __vowels = 'aeiouyæåø'
    __s_ending = 'bcdfghjlmnoprtvyz'
    __step1_suffixes = ('hetenes', 'hetene', 'hetens', 'heter', 'heten', 'endes', 'ande', 'ende', 'edes', 'enes', 'erte', 'ede', 'ane', 'ene', 'ens', 'ers', 'ets', 'het', 'ast', 'ert', 'en', 'ar', 'er', 'as', 'es', 'et', 'a', 'e', 's')
    __step2_suffixes = ('dt', 'vt')
    __step3_suffixes = ('hetslov', 'eleg', 'elig', 'elov', 'slov', 'leg', 'eig', 'lig', 'els', 'lov', 'ig')

    def stem(self, word):
        """
        Stem a Norwegian word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()
        if word in self.stopwords:
            return word
        r1 = self._r1_scandinavian(word, self.__vowels)
        for suffix in self.__step1_suffixes:
            if r1.endswith(suffix):
                if suffix in ('erte', 'ert'):
                    word = suffix_replace(word, suffix, 'er')
                    r1 = suffix_replace(r1, suffix, 'er')
                elif suffix == 's':
                    if word[-2] in self.__s_ending or (word[-2] == 'k' and word[-3] not in self.__vowels):
                        word = word[:-1]
                        r1 = r1[:-1]
                else:
                    word = word[:-len(suffix)]
                    r1 = r1[:-len(suffix)]
                break
        for suffix in self.__step2_suffixes:
            if r1.endswith(suffix):
                word = word[:-1]
                r1 = r1[:-1]
                break
        for suffix in self.__step3_suffixes:
            if r1.endswith(suffix):
                word = word[:-len(suffix)]
                break
        return word