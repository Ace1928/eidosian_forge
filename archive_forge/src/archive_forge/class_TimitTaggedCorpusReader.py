import os
from nltk.corpus.reader.api import *
from nltk.corpus.reader.timit import read_timit_block
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
from nltk.tokenize import *
class TimitTaggedCorpusReader(TaggedCorpusReader):
    """
    A corpus reader for tagged sentences that are included in the TIMIT corpus.
    """

    def __init__(self, *args, **kwargs):
        TaggedCorpusReader.__init__(self, *args, para_block_reader=read_timit_block, **kwargs)

    def paras(self):
        raise NotImplementedError('use sents() instead')

    def tagged_paras(self):
        raise NotImplementedError('use tagged_sents() instead')