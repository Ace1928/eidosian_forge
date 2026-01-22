import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _pprint_single_frame(self, vnframe, indent=''):
    """Returns pretty printed version of a single frame in a VerbNet class

        Returns a string containing a pretty-printed representation of
        the given frame.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        """
    frame_string = self._pprint_description_within_frame(vnframe, indent) + '\n'
    frame_string += self._pprint_example_within_frame(vnframe, indent + ' ') + '\n'
    frame_string += self._pprint_syntax_within_frame(vnframe, indent + '  Syntax: ') + '\n'
    frame_string += indent + '  Semantics:\n'
    frame_string += self._pprint_semantics_within_frame(vnframe, indent + '    ')
    return frame_string