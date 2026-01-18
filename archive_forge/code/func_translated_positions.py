import warnings
from collections import defaultdict
from math import log
def translated_positions(self):
    """
        List of positions in the source sentence of words already
        translated. The list is not sorted.

        :rtype: list(int)
        """
    translated_positions = []
    current_hypothesis = self
    while current_hypothesis.previous is not None:
        translated_span = current_hypothesis.src_phrase_span
        translated_positions.extend(range(translated_span[0], translated_span[1]))
        current_hypothesis = current_hypothesis.previous
    return translated_positions