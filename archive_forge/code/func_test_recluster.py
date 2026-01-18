import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def test_recluster(self):
    elda = EnsembleLda.load(datapath('ensemblelda'))
    loaded_cluster_model_results = deepcopy(elda.cluster_model.results)
    loaded_valid_clusters = deepcopy(elda.valid_clusters)
    loaded_stable_topics = deepcopy(elda.get_topics())
    elda.asymmetric_distance_matrix_outdated = True
    elda.recluster()
    self.assert_clustering_results_equal(elda.cluster_model.results, loaded_cluster_model_results)
    assert elda.valid_clusters == loaded_valid_clusters
    np.testing.assert_allclose(elda.get_topics(), loaded_stable_topics, rtol=RTOL)