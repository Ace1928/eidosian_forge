import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def _strip_tags(self, data):
    """
        Gets rid of all tags and newline characters from the given input

        :return: A cleaned-up version of the input string
        :rtype: str
        """
    try:
        "\n            # Look for boundary issues in markup. (Sometimes FEs are pluralized in definitions.)\n            m = re.search(r'\\w[<][^/]|[<][/][^>]+[>](s\\w|[a-rt-z0-9])', data)\n            if m:\n                print('Markup boundary:', data[max(0,m.start(0)-10):m.end(0)+10].replace('\\n',' '), file=sys.stderr)\n            "
        data = data.replace('<t>', '')
        data = data.replace('</t>', '')
        data = re.sub('<fex name="[^"]+">', '', data)
        data = data.replace('</fex>', '')
        data = data.replace('<fen>', '')
        data = data.replace('</fen>', '')
        data = data.replace('<m>', '')
        data = data.replace('</m>', '')
        data = data.replace('<ment>', '')
        data = data.replace('</ment>', '')
        data = data.replace('<ex>', "'")
        data = data.replace('</ex>', "'")
        data = data.replace('<gov>', '')
        data = data.replace('</gov>', '')
        data = data.replace('<x>', '')
        data = data.replace('</x>', '')
        data = data.replace('<def-root>', '')
        data = data.replace('</def-root>', '')
        data = data.replace('\n', ' ')
    except AttributeError:
        pass
    return data