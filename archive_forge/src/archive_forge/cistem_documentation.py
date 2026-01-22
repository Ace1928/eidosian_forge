import re
from typing import Tuple
from nltk.stem.api import StemmerI
Inner method for iteratively applying the code stemming regexes.
        This method receives a pre-processed variant of the word to be stemmed,
        or the word to be segmented, and returns a tuple of the word and the
        removed suffix.

        :param word: A pre-processed variant of the word that is to be stemmed.
        :type word: str
        :param upper: Whether the original word started with a capital letter.
        :type upper: bool
        :return: A tuple of the stemmed word and the removed suffix.
        :rtype: Tuple[str, str]
        