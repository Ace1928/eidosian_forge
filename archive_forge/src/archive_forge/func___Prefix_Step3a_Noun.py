import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Prefix_Step3a_Noun(self, token):
    for prefix in self.__prefix_step3a_noun:
        if token.startswith(prefix):
            if prefix in self.__articles_2len and len(token) > 4:
                token = token[len(prefix):]
                self.prefix_step3a_noun_success = True
                break
            if prefix in self.__articles_3len and len(token) > 5:
                token = token[len(prefix):]
                break
    return token