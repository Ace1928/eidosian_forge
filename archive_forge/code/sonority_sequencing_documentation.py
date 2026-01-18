import re
import warnings
from string import punctuation
from nltk.tokenize.api import TokenizerI
from nltk.util import ngrams

        Apply the SSP to return a list of syllables.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return syllable_list: Single word or token broken up into syllables.
        :rtype: list(str)
        