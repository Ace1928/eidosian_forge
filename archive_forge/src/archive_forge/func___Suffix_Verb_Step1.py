import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Suffix_Verb_Step1(self, token):
    for suffix in self.__suffix_verb_step1:
        if token.endswith(suffix):
            if suffix in self.__conjugation_suffix_verb_1 and len(token) >= 4:
                token = token[:-1]
                self.suffixes_verb_step1_success = True
                break
            if suffix in self.__conjugation_suffix_verb_2 and len(token) >= 5:
                token = token[:-2]
                self.suffixes_verb_step1_success = True
                break
            if suffix in self.__conjugation_suffix_verb_3 and len(token) >= 6:
                token = token[:-3]
                self.suffixes_verb_step1_success = True
                break
    return token