import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def test_persisting(self):
    elda = self.get_elda()
    elda_mem_unfriendly = self.get_elda_mem_unfriendly()
    fname = get_tmpfile('gensim_models_ensemblelda')
    elda.save(fname)
    loaded_elda = EnsembleLda.load(fname)
    elda_mem_unfriendly.save(fname)
    loaded_elda_mem_unfriendly = EnsembleLda.load(fname)
    assert loaded_elda.topic_model_class is None
    loaded_elda_representation = loaded_elda.generate_gensim_representation()
    assert loaded_elda.topic_model_class == LdaModel
    topics = loaded_elda_representation.get_topics()
    ttda = loaded_elda.ttda
    amatrix = loaded_elda.asymmetric_distance_matrix
    np.testing.assert_allclose(elda.get_topics(), topics, rtol=RTOL)
    np.testing.assert_allclose(elda.ttda, ttda, rtol=RTOL)
    np.testing.assert_allclose(elda.asymmetric_distance_matrix, amatrix, rtol=RTOL)
    expected_clustering_results = elda.cluster_model.results
    loaded_clustering_results = loaded_elda.cluster_model.results
    self.assert_clustering_results_equal(expected_clustering_results, loaded_clustering_results)
    loaded_elda_mem_unfriendly_representation = loaded_elda_mem_unfriendly.generate_gensim_representation()
    topics = loaded_elda_mem_unfriendly_representation.get_topics()
    np.testing.assert_allclose(elda.get_topics(), topics, rtol=RTOL)