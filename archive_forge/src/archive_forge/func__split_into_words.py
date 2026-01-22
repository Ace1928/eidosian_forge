import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def _split_into_words(sentences):
    """
        Splits multiple sentences into words and flattens the result

        Args:
          sentences: list of string

        Returns:
          A list of words (split by white space)
        """
    return list(itertools.chain(*[_.split() for _ in sentences]))