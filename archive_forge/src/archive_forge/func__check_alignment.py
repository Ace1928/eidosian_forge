import subprocess
from collections import namedtuple
def _check_alignment(num_words, num_mots, alignment):
    """
    Check whether the alignments are legal.

    :param num_words: the number of source language words
    :type num_words: int
    :param num_mots: the number of target language words
    :type num_mots: int
    :param alignment: alignment to be checked
    :type alignment: Alignment
    :raise IndexError: if alignment falls outside the sentence
    """
    assert type(alignment) is Alignment
    if not all((0 <= pair[0] < num_words for pair in alignment)):
        raise IndexError('Alignment is outside boundary of words')
    if not all((pair[1] is None or 0 <= pair[1] < num_mots for pair in alignment)):
        raise IndexError('Alignment is outside boundary of mots')