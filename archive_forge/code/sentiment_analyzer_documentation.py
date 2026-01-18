import sys
from collections import defaultdict
from nltk.classify.util import accuracy as eval_accuracy
from nltk.classify.util import apply_features
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import f_measure as eval_f_measure
from nltk.metrics import precision as eval_precision
from nltk.metrics import recall as eval_recall
from nltk.probability import FreqDist

        Evaluate and print classifier performance on the test set.

        :param test_set: A list of (tokens, label) tuples to use as gold set.
        :param classifier: a classifier instance (previously trained).
        :param accuracy: if `True`, evaluate classifier accuracy.
        :param f_measure: if `True`, evaluate classifier f_measure.
        :param precision: if `True`, evaluate classifier precision.
        :param recall: if `True`, evaluate classifier recall.
        :return: evaluation results.
        :rtype: dict(str): float
        