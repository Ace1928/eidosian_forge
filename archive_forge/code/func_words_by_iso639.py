import re
from collections import defaultdict, namedtuple
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.wordlist import WordListCorpusReader
from nltk.tokenize import line_tokenize
def words_by_iso639(self, iso63_code):
    """
        :return: a list of list(str)
        """
    fileids = [f'swadesh{self.swadesh_size}/{lang_code}.txt' for lang_code in self._macro_langauges[iso63_code]]
    return [concept.split('\t') for fileid in fileids for concept in self.words(fileid)]