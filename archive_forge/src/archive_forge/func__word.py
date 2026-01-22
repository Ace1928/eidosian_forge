from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import sinica_parse
def _word(self, sent):
    return WORD.findall(sent)