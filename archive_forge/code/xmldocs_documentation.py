import codecs
from xml.etree import ElementTree
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import *
from nltk.data import SeekableUnicodeStreamReader
from nltk.internals import ElementWrapper
from nltk.tokenize import WordPunctTokenizer

        Read from ``stream`` until we find at least one element that
        matches ``tagspec``, and return the result of applying
        ``elt_handler`` to each element found.
        