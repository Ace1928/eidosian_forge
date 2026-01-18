import subprocess
from collections import namedtuple
def translations_for(self, src_phrase):
    """
        Get the translations for a source language phrase

        :param src_phrase: Source language phrase of interest
        :type src_phrase: tuple(str)

        :return: A list of target language phrases that are translations
            of ``src_phrase``, ordered in decreasing order of
            likelihood. Each list element is a tuple of the target
            phrase and its log probability.
        :rtype: list(PhraseTableEntry)
        """
    return self.src_phrases[src_phrase]