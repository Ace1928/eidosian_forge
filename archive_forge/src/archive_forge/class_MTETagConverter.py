import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
class MTETagConverter:
    """
    Class for converting msd tags to universal tags, more conversion
    options are currently not implemented.
    """
    mapping_msd_universal = {'A': 'ADJ', 'S': 'ADP', 'R': 'ADV', 'C': 'CONJ', 'D': 'DET', 'N': 'NOUN', 'M': 'NUM', 'Q': 'PRT', 'P': 'PRON', 'V': 'VERB', '.': '.', '-': 'X'}

    @staticmethod
    def msd_to_universal(tag):
        """
        This function converts the annotation from the Multex-East to the universal tagset
        as described in Chapter 5 of the NLTK-Book

        Unknown Tags will be mapped to X. Punctuation marks are not supported in MSD tags, so
        """
        indicator = tag[0] if not tag[0] == '#' else tag[1]
        if not indicator in MTETagConverter.mapping_msd_universal:
            indicator = '-'
        return MTETagConverter.mapping_msd_universal[indicator]