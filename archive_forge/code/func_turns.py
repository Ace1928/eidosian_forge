import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
def turns(self):
    return StreamBackedCorpusView(self.abspath('tagged'), self._turns_block_reader)