from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def verb_t3(self, token):
    """
        stem the present suffixes
        """
    if len(token) > 5:
        for su3 in self.verb_suf3:
            if token.endswith(su3):
                return token[:-3]
    if len(token) > 4:
        for su2 in self.verb_suf2:
            if token.endswith(su2):
                return token[:-2]
    if len(token) > 3:
        for su1 in self.verb_suf1:
            if token.endswith(su1):
                return token[:-1]