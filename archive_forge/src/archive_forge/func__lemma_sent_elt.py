import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
@classmethod
def _lemma_sent_elt(cls, elt, context):
    return [cls._lemma_word_elt(w, None) for w in xpath(elt, '*', cls.ns)]