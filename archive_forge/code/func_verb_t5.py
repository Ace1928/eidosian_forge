from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def verb_t5(self, token):
    """
        stem the future prefixes
        """
    if len(token) > 4:
        for pr2 in self.verb_pr22:
            if token.startswith(pr2):
                return token[2:]
        for pr2 in self.verb_pr2:
            if token.startswith(pr2):
                return token[2:]
    return token