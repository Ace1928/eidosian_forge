import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _get_semantics_within_frame(self, vnframe):
    """Returns semantics within a single frame

        A utility function to retrieve semantics within a frame in VerbNet
        Members of the semantics dictionary:
        1) Predicate value
        2) Arguments

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        :return: semantics: semantics dictionary
        """
    semantics_within_single_frame = []
    for pred in vnframe.findall('SEMANTICS/PRED'):
        arguments = [{'type': arg.get('type'), 'value': arg.get('value')} for arg in pred.findall('ARGS/ARG')]
        semantics_within_single_frame.append({'predicate_value': pred.get('value'), 'arguments': arguments, 'negated': pred.get('bool') == '!'})
    return semantics_within_single_frame