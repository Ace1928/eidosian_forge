import nltk.data
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
def _read_para_block(self, stream):
    paras = []
    for para in self._para_block_reader(stream):
        paras.append([sent.split() for sent in para.splitlines()])
    return paras