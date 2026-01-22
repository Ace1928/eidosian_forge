import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
class ReviewLine:
    """
    A ReviewLine represents a sentence of the review, together with (optional)
    annotations of its features and notes about the reviewed item.
    """

    def __init__(self, sent, features=None, notes=None):
        self.sent = sent
        if features is None:
            self.features = []
        else:
            self.features = features
        if notes is None:
            self.notes = []
        else:
            self.notes = notes

    def __repr__(self):
        return 'ReviewLine(features={}, notes={}, sent={})'.format(self.features, self.notes, self.sent)