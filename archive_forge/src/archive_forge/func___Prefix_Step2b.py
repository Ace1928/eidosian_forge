import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Prefix_Step2b(self, token):
    for prefix in self.__prefix_step2b:
        if token.startswith(prefix) and len(token) > 3:
            if token[:2] not in self.__prefixes1:
                token = token[len(prefix):]
                break
    return token