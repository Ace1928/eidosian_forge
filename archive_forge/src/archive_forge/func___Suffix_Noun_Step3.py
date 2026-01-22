import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Suffix_Noun_Step3(self, token):
    for suffix in self.__suffix_noun_step3:
        if token.endswith(suffix) and len(token) >= 3:
            token = token[:-1]
            break
    return token