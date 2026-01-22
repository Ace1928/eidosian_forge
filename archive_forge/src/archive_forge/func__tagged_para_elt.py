import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
@classmethod
def _tagged_para_elt(cls, elt, context):
    return list(filter(lambda x: x is not None, [cls._tagged_sent_elt(s, None) for s in xpath(elt, '*', cls.ns)]))