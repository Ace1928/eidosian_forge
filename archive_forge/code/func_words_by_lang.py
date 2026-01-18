import re
from collections import defaultdict, namedtuple
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.wordlist import WordListCorpusReader
from nltk.tokenize import line_tokenize
def words_by_lang(self, lang_code):
    """
        :return: a list of list(str)
        """
    fileid = f'swadesh{self.swadesh_size}/{lang_code}.txt'
    return [concept.split('\t') for concept in self.words(fileid)]