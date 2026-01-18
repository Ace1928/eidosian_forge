import pytest
from nltk import config_megam
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.corpus import rte as rte_corpus
def test_rte_feature_extraction(self):
    pairs = rte_corpus.pairs(['rte1_dev.xml'])[:6]
    test_output = [f'{key:<15} => {rte_features(pair)[key]}' for pair in pairs for key in sorted(rte_features(pair))]
    expected_output = expected_from_rte_feature_extration.strip().split('\n')
    expected_output = list(filter(None, expected_output))
    assert test_output == expected_output