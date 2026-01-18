import numpy as np
from sklearn.metrics import consensus_score
from sklearn.metrics.cluster._bicluster import _jaccard
from sklearn.utils._testing import assert_almost_equal
Different number of biclusters in A and B