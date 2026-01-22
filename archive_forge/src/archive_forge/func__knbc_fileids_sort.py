import re
from nltk.corpus.reader.api import CorpusReader, SyntaxCorpusReader
from nltk.corpus.reader.util import (
from nltk.parse import DependencyGraph
def _knbc_fileids_sort(x):
    cells = x.split('-')
    return (cells[0], int(cells[1]), int(cells[2]), int(cells[3]))