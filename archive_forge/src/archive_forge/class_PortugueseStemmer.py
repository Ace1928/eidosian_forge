import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
class PortugueseStemmer(_StandardStemmer):
    """
    The Portuguese Snowball stemmer.

    :cvar __vowels: The Portuguese vowels.
    :type __vowels: unicode
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step4_suffixes: Suffixes to be deleted in step 4 of the algorithm.
    :type __step4_suffixes: tuple
    :note: A detailed description of the Portuguese
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/portuguese/stemmer.html

    """
    __vowels = 'aeiouáéíóúâêô'
    __step1_suffixes = ('amentos', 'imentos', 'uço~es', 'amento', 'imento', 'adoras', 'adores', 'aço~es', 'logias', 'ências', 'amente', 'idades', 'anças', 'ismos', 'istas', 'adora', 'aça~o', 'antes', 'ância', 'logia', 'uça~o', 'ência', 'mente', 'idade', 'ança', 'ezas', 'icos', 'icas', 'ismo', 'ável', 'ível', 'ista', 'osos', 'osas', 'ador', 'ante', 'ivas', 'ivos', 'iras', 'eza', 'ico', 'ica', 'oso', 'osa', 'iva', 'ivo', 'ira')
    __step2_suffixes = ('aríamos', 'eríamos', 'iríamos', 'ássemos', 'êssemos', 'íssemos', 'aríeis', 'eríeis', 'iríeis', 'ásseis', 'ésseis', 'ísseis', 'áramos', 'éramos', 'íramos', 'ávamos', 'aremos', 'eremos', 'iremos', 'ariam', 'eriam', 'iriam', 'assem', 'essem', 'issem', 'ara~o', 'era~o', 'ira~o', 'arias', 'erias', 'irias', 'ardes', 'erdes', 'irdes', 'asses', 'esses', 'isses', 'astes', 'estes', 'istes', 'áreis', 'areis', 'éreis', 'ereis', 'íreis', 'ireis', 'áveis', 'íamos', 'armos', 'ermos', 'irmos', 'aria', 'eria', 'iria', 'asse', 'esse', 'isse', 'aste', 'este', 'iste', 'arei', 'erei', 'irei', 'aram', 'eram', 'iram', 'avam', 'arem', 'erem', 'irem', 'ando', 'endo', 'indo', 'adas', 'idas', 'arás', 'aras', 'erás', 'eras', 'irás', 'avas', 'ares', 'eres', 'ires', 'íeis', 'ados', 'idos', 'ámos', 'amos', 'emos', 'imos', 'iras', 'ada', 'ida', 'ará', 'ara', 'erá', 'era', 'irá', 'ava', 'iam', 'ado', 'ido', 'ias', 'ais', 'eis', 'ira', 'ia', 'ei', 'am', 'em', 'ar', 'er', 'ir', 'as', 'es', 'is', 'eu', 'iu', 'ou')
    __step4_suffixes = ('os', 'a', 'i', 'o', 'á', 'í', 'ó')

    def stem(self, word):
        """
        Stem a Portuguese word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()
        if word in self.stopwords:
            return word
        step1_success = False
        step2_success = False
        word = word.replace('ã', 'a~').replace('õ', 'o~').replace('qü', 'qu').replace('gü', 'gu')
        r1, r2 = self._r1r2_standard(word, self.__vowels)
        rv = self._rv_standard(word, self.__vowels)
        for suffix in self.__step1_suffixes:
            if word.endswith(suffix):
                if suffix == 'amente' and r1.endswith(suffix):
                    step1_success = True
                    word = word[:-6]
                    r2 = r2[:-6]
                    rv = rv[:-6]
                    if r2.endswith('iv'):
                        word = word[:-2]
                        r2 = r2[:-2]
                        rv = rv[:-2]
                        if r2.endswith('at'):
                            word = word[:-2]
                            rv = rv[:-2]
                    elif r2.endswith(('os', 'ic', 'ad')):
                        word = word[:-2]
                        rv = rv[:-2]
                elif suffix in ('ira', 'iras') and rv.endswith(suffix) and (word[-len(suffix) - 1:-len(suffix)] == 'e'):
                    step1_success = True
                    word = suffix_replace(word, suffix, 'ir')
                    rv = suffix_replace(rv, suffix, 'ir')
                elif r2.endswith(suffix):
                    step1_success = True
                    if suffix in ('logia', 'logias'):
                        word = suffix_replace(word, suffix, 'log')
                        rv = suffix_replace(rv, suffix, 'log')
                    elif suffix in ('uça~o', 'uço~es'):
                        word = suffix_replace(word, suffix, 'u')
                        rv = suffix_replace(rv, suffix, 'u')
                    elif suffix in ('ência', 'ências'):
                        word = suffix_replace(word, suffix, 'ente')
                        rv = suffix_replace(rv, suffix, 'ente')
                    elif suffix == 'mente':
                        word = word[:-5]
                        r2 = r2[:-5]
                        rv = rv[:-5]
                        if r2.endswith(('ante', 'avel', 'ivel')):
                            word = word[:-4]
                            rv = rv[:-4]
                    elif suffix in ('idade', 'idades'):
                        word = word[:-len(suffix)]
                        r2 = r2[:-len(suffix)]
                        rv = rv[:-len(suffix)]
                        if r2.endswith(('ic', 'iv')):
                            word = word[:-2]
                            rv = rv[:-2]
                        elif r2.endswith('abil'):
                            word = word[:-4]
                            rv = rv[:-4]
                    elif suffix in ('iva', 'ivo', 'ivas', 'ivos'):
                        word = word[:-len(suffix)]
                        r2 = r2[:-len(suffix)]
                        rv = rv[:-len(suffix)]
                        if r2.endswith('at'):
                            word = word[:-2]
                            rv = rv[:-2]
                    else:
                        word = word[:-len(suffix)]
                        rv = rv[:-len(suffix)]
                break
        if not step1_success:
            for suffix in self.__step2_suffixes:
                if rv.endswith(suffix):
                    step2_success = True
                    word = word[:-len(suffix)]
                    rv = rv[:-len(suffix)]
                    break
        if step1_success or step2_success:
            if rv.endswith('i') and word[-2] == 'c':
                word = word[:-1]
                rv = rv[:-1]
        if not step1_success and (not step2_success):
            for suffix in self.__step4_suffixes:
                if rv.endswith(suffix):
                    word = word[:-len(suffix)]
                    rv = rv[:-len(suffix)]
                    break
        if rv.endswith(('e', 'é', 'ê')):
            word = word[:-1]
            rv = rv[:-1]
            if word.endswith('gu') and rv.endswith('u') or (word.endswith('ci') and rv.endswith('i')):
                word = word[:-1]
        elif word.endswith('ç'):
            word = suffix_replace(word, 'ç', 'c')
        word = word.replace('a~', 'ã').replace('o~', 'õ')
        return word