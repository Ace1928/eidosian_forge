from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import manifold, neighbors
from sklearn.datasets import make_blobs
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
from sklearn.utils._testing import (
Check get_feature_names_out for LocallyLinearEmbedding.