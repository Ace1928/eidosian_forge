import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
def _iter_hypernym_lists(self):
    """
        :return: An iterator over ``Synset`` objects that are either proper
        hypernyms or instance of hypernyms of the synset.
        """
    todo = [self]
    seen = set()
    while todo:
        for synset in todo:
            seen.add(synset)
        yield todo
        todo = [hypernym for synset in todo for hypernym in synset.hypernyms() + synset.instance_hypernyms() if hypernym not in seen]