import re
from collections import defaultdict
from functools import reduce
from nltk.corpus.reader import CorpusReader
@staticmethod
def __defaultdict_factory():
    """Factory for creating defaultdict of defaultdict(dict)s"""
    return defaultdict(dict)