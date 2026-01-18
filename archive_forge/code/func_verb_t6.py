from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def verb_t6(self, token):
    """
        stem the order prefixes
        """
    if len(token) > 4:
        for pr3 in self.verb_pr33:
            if token.startswith(pr3):
                return token[2:]
    return token