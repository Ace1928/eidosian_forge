import os
import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.corpus.reader.util import *
from nltk.tokenize import RegexpTokenizer
class YCOETaggedCorpusReader(TaggedCorpusReader):

    def __init__(self, root, items, encoding='utf8'):
        gaps_re = '(?u)(?<=/\\.)\\s+|\\s*\\S*_CODE\\s*|\\s*\\S*_ID\\s*'
        sent_tokenizer = RegexpTokenizer(gaps_re, gaps=True)
        TaggedCorpusReader.__init__(self, root, items, sep='_', sent_tokenizer=sent_tokenizer)