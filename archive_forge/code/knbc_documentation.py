import re
from nltk.corpus.reader.api import CorpusReader, SyntaxCorpusReader
from nltk.corpus.reader.util import (
from nltk.parse import DependencyGraph

        Initialize KNBCorpusReader
        morphs2str is a function to convert morphlist to str for tree representation
        for _parse()
        