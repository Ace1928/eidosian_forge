import re
from nltk.stem.api import StemmerI
def _ends_double_consonant(self, word):
    """Implements condition *d from the paper

        Returns True if word ends with a double consonant
        """
    return len(word) >= 2 and word[-1] == word[-2] and self._is_consonant(word, len(word) - 1)