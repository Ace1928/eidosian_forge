import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
Returns a pretty printed version of semantics within frame in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet frame semantics.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        