import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
@classmethod
def _lemma_word_elt(cls, elt, context):
    if 'lemma' not in elt.attrib:
        return (elt.text, '')
    else:
        return (elt.text, elt.attrib['lemma'])