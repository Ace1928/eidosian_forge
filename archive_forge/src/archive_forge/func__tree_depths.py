import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
def _tree_depths(A, root):
    if not graphs.Graph(A=A).is_connected():
        raise ValueError('Graph is not connected')
    N = np.shape(A)[0]
    assigned = root - 1
    depths = np.zeros(N)
    parents = np.zeros(N)
    next_to_expand = np.array([root])
    current_depth = 1
    while len(assigned) < N:
        new_entries_whole_round = []
        for i in range(len(next_to_expand)):
            neighbors = np.where(A[next_to_expand[i]])[0]
            new_entries = np.setdiff1d(neighbors, assigned)
            parents[new_entries] = next_to_expand[i]
            depths[new_entries] = current_depth
            assigned = np.concatenate((assigned, new_entries))
            new_entries_whole_round = np.concatenate((new_entries_whole_round, new_entries))
        current_depth = current_depth + 1
        next_to_expand = new_entries_whole_round
    return (depths, parents)