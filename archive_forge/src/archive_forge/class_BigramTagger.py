import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI
@jsontags.register_tag
class BigramTagger(NgramTagger):
    """
    A tagger that chooses a token's tag based its word string and on
    the preceding words' tag.  In particular, a tuple consisting
    of the previous tag and the word is looked up in a table, and
    the corresponding tag is returned.

    :param train: The corpus of training data, a list of tagged sentences
    :type train: list(list(tuple(str, str)))
    :param model: The tagger model
    :type model: dict
    :param backoff: Another tagger which this tagger will consult when it is
        unable to tag a word
    :type backoff: TaggerI
    :param cutoff: The number of instances of training data the tagger must see
        in order not to use the backoff tagger
    :type cutoff: int
    """
    json_tag = 'nltk.tag.sequential.BigramTagger'

    def __init__(self, train=None, model=None, backoff=None, cutoff=0, verbose=False):
        super().__init__(2, train, model, backoff, cutoff, verbose)