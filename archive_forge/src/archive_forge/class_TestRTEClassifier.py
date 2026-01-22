import pytest
from nltk import config_megam
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.corpus import rte as rte_corpus
class TestRTEClassifier:

    def test_rte_feature_extraction(self):
        pairs = rte_corpus.pairs(['rte1_dev.xml'])[:6]
        test_output = [f'{key:<15} => {rte_features(pair)[key]}' for pair in pairs for key in sorted(rte_features(pair))]
        expected_output = expected_from_rte_feature_extration.strip().split('\n')
        expected_output = list(filter(None, expected_output))
        assert test_output == expected_output

    def test_feature_extractor_object(self):
        rtepair = rte_corpus.pairs(['rte3_dev.xml'])[33]
        extractor = RTEFeatureExtractor(rtepair)
        assert extractor.hyp_words == {'member', 'China', 'SCO.'}
        assert extractor.overlap('word') == set()
        assert extractor.overlap('ne') == {'China'}
        assert extractor.hyp_extra('word') == {'member'}

    def test_rte_classification_without_megam(self):
        clf = rte_classifier('IIS', sample_N=100)
        clf = rte_classifier('GIS', sample_N=100)

    def test_rte_classification_with_megam(self):
        try:
            config_megam()
        except (LookupError, AttributeError) as e:
            pytest.skip('Skipping tests with dependencies on MEGAM')
        clf = rte_classifier('megam', sample_N=100)