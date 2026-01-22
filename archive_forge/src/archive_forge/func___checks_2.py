import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __checks_2(self, token):
    for suffix in self.__checks2:
        if token.endswith(suffix):
            if suffix == 'ة' and len(token) > 2:
                self.is_noun = True
                self.is_verb = False
                break
            if suffix == 'ات' and len(token) > 3:
                self.is_noun = True
                self.is_verb = False
                break