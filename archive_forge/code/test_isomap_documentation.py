import math
from itertools import product
import numpy as np
import pytest
from scipy.sparse import rand as sparse_rand
from sklearn import clone, datasets, manifold, neighbors, pipeline, preprocessing
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
Check get_feature_names_out for Isomap.