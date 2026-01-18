import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal

This is testing the equivalence between some estimators with internal nearest
neighbors computations, and the corresponding pipeline versions with
KNeighborsTransformer or RadiusNeighborsTransformer to precompute the
neighbors.
