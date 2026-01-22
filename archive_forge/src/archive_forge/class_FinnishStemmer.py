import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
class FinnishStemmer(_StandardStemmer):
    """
    The Finnish Snowball stemmer.

    :cvar __vowels: The Finnish vowels.
    :type __vowels: unicode
    :cvar __restricted_vowels: A subset of the Finnish vowels.
    :type __restricted_vowels: unicode
    :cvar __long_vowels: The Finnish vowels in their long forms.
    :type __long_vowels: tuple
    :cvar __consonants: The Finnish consonants.
    :type __consonants: unicode
    :cvar __double_consonants: The Finnish double consonants.
    :type __double_consonants: tuple
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :cvar __step4_suffixes: Suffixes to be deleted in step 4 of the algorithm.
    :type __step4_suffixes: tuple
    :note: A detailed description of the Finnish
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/finnish/stemmer.html
    """
    __vowels = 'aeiouyäö'
    __restricted_vowels = 'aeiouäö'
    __long_vowels = ('aa', 'ee', 'ii', 'oo', 'uu', 'ää', 'öö')
    __consonants = 'bcdfghjklmnpqrstvwxz'
    __double_consonants = ('bb', 'cc', 'dd', 'ff', 'gg', 'hh', 'jj', 'kk', 'll', 'mm', 'nn', 'pp', 'qq', 'rr', 'ss', 'tt', 'vv', 'ww', 'xx', 'zz')
    __step1_suffixes = ('kaan', 'kään', 'sti', 'kin', 'han', 'hän', 'ko', 'kö', 'pa', 'pä')
    __step2_suffixes = ('nsa', 'nsä', 'mme', 'nne', 'si', 'ni', 'an', 'än', 'en')
    __step3_suffixes = ('siin', 'tten', 'seen', 'han', 'hen', 'hin', 'hon', 'hän', 'hön', 'den', 'tta', 'ttä', 'ssa', 'ssä', 'sta', 'stä', 'lla', 'llä', 'lta', 'ltä', 'lle', 'ksi', 'ine', 'ta', 'tä', 'na', 'nä', 'a', 'ä', 'n')
    __step4_suffixes = ('impi', 'impa', 'impä', 'immi', 'imma', 'immä', 'mpi', 'mpa', 'mpä', 'mmi', 'mma', 'mmä', 'eja', 'ejä')

    def stem(self, word):
        """
        Stem a Finnish word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()
        if word in self.stopwords:
            return word
        step3_success = False
        r1, r2 = self._r1r2_standard(word, self.__vowels)
        for suffix in self.__step1_suffixes:
            if r1.endswith(suffix):
                if suffix == 'sti':
                    if suffix in r2:
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                elif word[-len(suffix) - 1] in 'ntaeiouyäö':
                    word = word[:-len(suffix)]
                    r1 = r1[:-len(suffix)]
                    r2 = r2[:-len(suffix)]
                break
        for suffix in self.__step2_suffixes:
            if r1.endswith(suffix):
                if suffix == 'si':
                    if word[-3] != 'k':
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                elif suffix == 'ni':
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]
                    if word.endswith('kse'):
                        word = suffix_replace(word, 'kse', 'ksi')
                    if r1.endswith('kse'):
                        r1 = suffix_replace(r1, 'kse', 'ksi')
                    if r2.endswith('kse'):
                        r2 = suffix_replace(r2, 'kse', 'ksi')
                elif suffix == 'an':
                    if word[-4:-2] in ('ta', 'na') or word[-5:-2] in ('ssa', 'sta', 'lla', 'lta'):
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                elif suffix == 'än':
                    if word[-4:-2] in ('tä', 'nä') or word[-5:-2] in ('ssä', 'stä', 'llä', 'ltä'):
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                elif suffix == 'en':
                    if word[-5:-2] in ('lle', 'ine'):
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                else:
                    word = word[:-3]
                    r1 = r1[:-3]
                    r2 = r2[:-3]
                break
        for suffix in self.__step3_suffixes:
            if r1.endswith(suffix):
                if suffix in ('han', 'hen', 'hin', 'hon', 'hän', 'hön'):
                    if suffix == 'han' and word[-4] == 'a' or (suffix == 'hen' and word[-4] == 'e') or (suffix == 'hin' and word[-4] == 'i') or (suffix == 'hon' and word[-4] == 'o') or (suffix == 'hän' and word[-4] == 'ä') or (suffix == 'hön' and word[-4] == 'ö'):
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                        step3_success = True
                elif suffix in ('siin', 'den', 'tten'):
                    if word[-len(suffix) - 1] == 'i' and word[-len(suffix) - 2] in self.__restricted_vowels:
                        word = word[:-len(suffix)]
                        r1 = r1[:-len(suffix)]
                        r2 = r2[:-len(suffix)]
                        step3_success = True
                    else:
                        continue
                elif suffix == 'seen':
                    if word[-6:-4] in self.__long_vowels:
                        word = word[:-4]
                        r1 = r1[:-4]
                        r2 = r2[:-4]
                        step3_success = True
                    else:
                        continue
                elif suffix in ('a', 'ä'):
                    if word[-2] in self.__vowels and word[-3] in self.__consonants:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                        step3_success = True
                elif suffix in ('tta', 'ttä'):
                    if word[-4] == 'e':
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                        step3_success = True
                elif suffix == 'n':
                    word = word[:-1]
                    r1 = r1[:-1]
                    r2 = r2[:-1]
                    step3_success = True
                    if word[-2:] == 'ie' or word[-2:] in self.__long_vowels:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                else:
                    word = word[:-len(suffix)]
                    r1 = r1[:-len(suffix)]
                    r2 = r2[:-len(suffix)]
                    step3_success = True
                break
        for suffix in self.__step4_suffixes:
            if r2.endswith(suffix):
                if suffix in ('mpi', 'mpa', 'mpä', 'mmi', 'mma', 'mmä'):
                    if word[-5:-3] != 'po':
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                else:
                    word = word[:-len(suffix)]
                    r1 = r1[:-len(suffix)]
                    r2 = r2[:-len(suffix)]
                break
        if step3_success and len(r1) >= 1 and (r1[-1] in 'ij'):
            word = word[:-1]
            r1 = r1[:-1]
        elif not step3_success and len(r1) >= 2 and (r1[-1] == 't') and (r1[-2] in self.__vowels):
            word = word[:-1]
            r1 = r1[:-1]
            r2 = r2[:-1]
            if r2.endswith('imma'):
                word = word[:-4]
                r1 = r1[:-4]
            elif r2.endswith('mma') and r2[-5:-3] != 'po':
                word = word[:-3]
                r1 = r1[:-3]
        if r1[-2:] in self.__long_vowels:
            word = word[:-1]
            r1 = r1[:-1]
        if len(r1) >= 2 and r1[-2] in self.__consonants and (r1[-1] in 'aäei'):
            word = word[:-1]
            r1 = r1[:-1]
        if r1.endswith(('oj', 'uj')):
            word = word[:-1]
            r1 = r1[:-1]
        if r1.endswith('jo'):
            word = word[:-1]
            r1 = r1[:-1]
        for i in range(1, len(word)):
            if word[-i] in self.__vowels:
                continue
            else:
                if i == 1:
                    if word[-i - 1:] in self.__double_consonants:
                        word = word[:-1]
                elif word[-i - 1:-i + 1] in self.__double_consonants:
                    word = ''.join((word[:-i], word[-i + 1:]))
                break
        return word