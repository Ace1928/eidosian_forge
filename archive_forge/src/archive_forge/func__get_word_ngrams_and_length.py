import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def _get_word_ngrams_and_length(n, sentences):
    """
        Calculates word n-grams for multiple sentences.

        Args:
          n: wich n-grams to calculate
          sentences: list of string

        Returns:
          A set of n-grams, their frequency and #n-grams in sentences
        """
    assert len(sentences) > 0
    assert n > 0
    tokens = Rouge._split_into_words(sentences)
    return (Rouge._get_ngrams(n, tokens), tokens, len(tokens) - (n - 1))