import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
class RomanianStemmer(_StandardStemmer):
    """
    The Romanian Snowball stemmer.

    :cvar __vowels: The Romanian vowels.
    :type __vowels: unicode
    :cvar __step0_suffixes: Suffixes to be deleted in step 0 of the algorithm.
    :type __step0_suffixes: tuple
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :note: A detailed description of the Romanian
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/romanian/stemmer.html

    """
    __vowels = 'aeiouăâî'
    __step0_suffixes = ('iilor', 'ului', 'elor', 'iile', 'ilor', 'atei', 'aţie', 'aţia', 'aua', 'ele', 'iua', 'iei', 'ile', 'ul', 'ea', 'ii')
    __step1_suffixes = ('abilitate', 'abilitati', 'abilităţi', 'ibilitate', 'abilităi', 'ivitate', 'ivitati', 'ivităţi', 'icitate', 'icitati', 'icităţi', 'icatori', 'ivităi', 'icităi', 'icator', 'aţiune', 'atoare', 'ătoare', 'iţiune', 'itoare', 'iciva', 'icive', 'icivi', 'icivă', 'icala', 'icale', 'icali', 'icală', 'ativa', 'ative', 'ativi', 'ativă', 'atori', 'ători', 'itiva', 'itive', 'itivi', 'itivă', 'itori', 'iciv', 'ical', 'ativ', 'ator', 'ător', 'itiv', 'itor')
    __step2_suffixes = ('abila', 'abile', 'abili', 'abilă', 'ibila', 'ibile', 'ibili', 'ibilă', 'atori', 'itate', 'itati', 'ităţi', 'abil', 'ibil', 'oasa', 'oasă', 'oase', 'anta', 'ante', 'anti', 'antă', 'ator', 'ităi', 'iune', 'iuni', 'isme', 'ista', 'iste', 'isti', 'istă', 'işti', 'ata', 'ată', 'ati', 'ate', 'uta', 'ută', 'uti', 'ute', 'ita', 'ită', 'iti', 'ite', 'ica', 'ice', 'ici', 'ică', 'osi', 'oşi', 'ant', 'iva', 'ive', 'ivi', 'ivă', 'ism', 'ist', 'at', 'ut', 'it', 'ic', 'os', 'iv')
    __step3_suffixes = ('seserăţi', 'aserăţi', 'iserăţi', 'âserăţi', 'userăţi', 'seserăm', 'aserăm', 'iserăm', 'âserăm', 'userăm', 'serăţi', 'seseşi', 'seseră', 'ească', 'arăţi', 'urăţi', 'irăţi', 'ârăţi', 'aseşi', 'aseră', 'iseşi', 'iseră', 'âseşi', 'âseră', 'useşi', 'useră', 'serăm', 'sesem', 'indu', 'ându', 'ează', 'eşti', 'eşte', 'ăşti', 'ăşte', 'eaţi', 'iaţi', 'arăm', 'urăm', 'irăm', 'ârăm', 'asem', 'isem', 'âsem', 'usem', 'seşi', 'seră', 'sese', 'are', 'ere', 'ire', 'âre', 'ind', 'ând', 'eze', 'ezi', 'esc', 'ăsc', 'eam', 'eai', 'eau', 'iam', 'iai', 'iau', 'aşi', 'ară', 'uşi', 'ură', 'işi', 'iră', 'âşi', 'âră', 'ase', 'ise', 'âse', 'use', 'aţi', 'eţi', 'iţi', 'âţi', 'sei', 'ez', 'am', 'ai', 'au', 'ea', 'ia', 'ui', 'âi', 'ăm', 'em', 'im', 'âm', 'se')

    def stem(self, word):
        """
        Stem a Romanian word and return the stemmed form.

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
        for i in range(1, len(word) - 1):
            if word[i - 1] in self.__vowels and word[i + 1] in self.__vowels:
                if word[i] == 'u':
                    word = ''.join((word[:i], 'U', word[i + 1:]))
                elif word[i] == 'i':
                    word = ''.join((word[:i], 'I', word[i + 1:]))
        r1, r2 = self._r1r2_standard(word, self.__vowels)
        rv = self._rv_standard(word, self.__vowels)
        for suffix in self.__step0_suffixes:
            if word.endswith(suffix):
                if suffix in r1:
                    if suffix in ('ul', 'ului'):
                        word = word[:-len(suffix)]
                        if suffix in rv:
                            rv = rv[:-len(suffix)]
                        else:
                            rv = ''
                    elif suffix == 'aua' or suffix == 'atei' or (suffix == 'ile' and word[-5:-3] != 'ab'):
                        word = word[:-2]
                    elif suffix in ('ea', 'ele', 'elor'):
                        word = suffix_replace(word, suffix, 'e')
                        if suffix in rv:
                            rv = suffix_replace(rv, suffix, 'e')
                        else:
                            rv = ''
                    elif suffix in ('ii', 'iua', 'iei', 'iile', 'iilor', 'ilor'):
                        word = suffix_replace(word, suffix, 'i')
                        if suffix in rv:
                            rv = suffix_replace(rv, suffix, 'i')
                        else:
                            rv = ''
                    elif suffix in ('aţie', 'aţia'):
                        word = word[:-1]
                break
        while True:
            replacement_done = False
            for suffix in self.__step1_suffixes:
                if word.endswith(suffix):
                    if suffix in r1:
                        step1_success = True
                        replacement_done = True
                        if suffix in ('abilitate', 'abilitati', 'abilităi', 'abilităţi'):
                            word = suffix_replace(word, suffix, 'abil')
                        elif suffix == 'ibilitate':
                            word = word[:-5]
                        elif suffix in ('ivitate', 'ivitati', 'ivităi', 'ivităţi'):
                            word = suffix_replace(word, suffix, 'iv')
                        elif suffix in ('icitate', 'icitati', 'icităi', 'icităţi', 'icator', 'icatori', 'iciv', 'iciva', 'icive', 'icivi', 'icivă', 'ical', 'icala', 'icale', 'icali', 'icală'):
                            word = suffix_replace(word, suffix, 'ic')
                        elif suffix in ('ativ', 'ativa', 'ative', 'ativi', 'ativă', 'aţiune', 'atoare', 'ator', 'atori', 'ătoare', 'ător', 'ători'):
                            word = suffix_replace(word, suffix, 'at')
                            if suffix in r2:
                                r2 = suffix_replace(r2, suffix, 'at')
                        elif suffix in ('itiv', 'itiva', 'itive', 'itivi', 'itivă', 'iţiune', 'itoare', 'itor', 'itori'):
                            word = suffix_replace(word, suffix, 'it')
                            if suffix in r2:
                                r2 = suffix_replace(r2, suffix, 'it')
                    else:
                        step1_success = False
                    break
            if not replacement_done:
                break
        for suffix in self.__step2_suffixes:
            if word.endswith(suffix):
                if suffix in r2:
                    step2_success = True
                    if suffix in ('iune', 'iuni'):
                        if word[-5] == 'ţ':
                            word = ''.join((word[:-5], 't'))
                    elif suffix in ('ism', 'isme', 'ist', 'ista', 'iste', 'isti', 'istă', 'işti'):
                        word = suffix_replace(word, suffix, 'ist')
                    else:
                        word = word[:-len(suffix)]
                break
        if not step1_success and (not step2_success):
            for suffix in self.__step3_suffixes:
                if word.endswith(suffix):
                    if suffix in rv:
                        if suffix in ('seserăţi', 'seserăm', 'serăţi', 'seseşi', 'seseră', 'serăm', 'sesem', 'seşi', 'seră', 'sese', 'aţi', 'eţi', 'iţi', 'âţi', 'sei', 'ăm', 'em', 'im', 'âm', 'se'):
                            word = word[:-len(suffix)]
                            rv = rv[:-len(suffix)]
                        elif not rv.startswith(suffix) and rv[rv.index(suffix) - 1] not in 'aeioăâî':
                            word = word[:-len(suffix)]
                        break
        for suffix in ('ie', 'a', 'e', 'i', 'ă'):
            if word.endswith(suffix):
                if suffix in rv:
                    word = word[:-len(suffix)]
                break
        word = word.replace('I', 'i').replace('U', 'u')
        return word