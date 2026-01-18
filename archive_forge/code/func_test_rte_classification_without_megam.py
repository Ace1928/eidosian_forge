import pytest
from nltk import config_megam
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.corpus import rte as rte_corpus
def test_rte_classification_without_megam(self):
    clf = rte_classifier('IIS', sample_N=100)
    clf = rte_classifier('GIS', sample_N=100)