import re
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
class SensevalInstance:

    def __init__(self, word, position, context, senses):
        self.word = word
        self.senses = tuple(senses)
        self.position = position
        self.context = context

    def __repr__(self):
        return 'SensevalInstance(word=%r, position=%r, context=%r, senses=%r)' % (self.word, self.position, self.context, self.senses)