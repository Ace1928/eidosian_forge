import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __normalize_post(self, token):
    for hamza in self.__last_hamzat:
        if token.endswith(hamza):
            token = suffix_replace(token, hamza, 'ء')
            break
    token = self.__initial_hamzat.sub('ا', token)
    token = self.__waw_hamza.sub('و', token)
    token = self.__yeh_hamza.sub('ي', token)
    token = self.__alefat.sub('ا', token)
    return token