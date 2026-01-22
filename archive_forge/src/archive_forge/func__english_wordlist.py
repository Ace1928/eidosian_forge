import os
import pickle
import re
from xml.etree import ElementTree as ET
from nltk.tag import ClassifierBasedTagger, pos_tag
from nltk.chunk.api import ChunkParserI
from nltk.chunk.util import ChunkScore
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
def _english_wordlist(self):
    try:
        wl = self._en_wordlist
    except AttributeError:
        from nltk.corpus import words
        self._en_wordlist = set(words.words('en-basic'))
        wl = self._en_wordlist
    return wl