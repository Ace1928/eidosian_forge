import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace

        Stem a Swedish word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        