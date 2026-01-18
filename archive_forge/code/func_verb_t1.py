from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def verb_t1(self, token):
    """
        stem the present prefixes and suffixes
        """
    if len(token) > 5 and token.startswith('ت'):
        for s2 in self.pl_si2:
            if token.endswith(s2):
                return token[1:-2]
    if len(token) > 5 and token.startswith('ي'):
        for s2 in self.verb_su2:
            if token.endswith(s2):
                return token[1:-2]
    if len(token) > 4 and token.startswith('ا'):
        if len(token) > 5 and token.endswith('وا'):
            return token[1:-2]
        if token.endswith('ي'):
            return token[1:-1]
        if token.endswith('ا'):
            return token[1:-1]
        if token.endswith('ن'):
            return token[1:-1]
    if len(token) > 4 and token.startswith('ي') and token.endswith('ن'):
        return token[1:-1]
    if len(token) > 4 and token.startswith('ت') and token.endswith('ن'):
        return token[1:-1]