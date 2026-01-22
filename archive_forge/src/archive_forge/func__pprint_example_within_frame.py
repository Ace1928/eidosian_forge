import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _pprint_example_within_frame(self, vnframe, indent=''):
    """Returns pretty printed version of example within frame in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet frame example.

        :param vnframe: An ElementTree containing the xml contents of
            a Verbnet frame.
        """
    if vnframe['example']:
        return indent + ' Example: ' + vnframe['example']