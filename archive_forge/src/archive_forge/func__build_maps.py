import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.common.dependencies import attempt_import
def _build_maps(self, list_of_matrices):
    """
        This method creates the maps that are used in condensed_sum.
        These maps remain valid as long as the nonzero structure of
        the individual matrices does not change
        """
    nz_tuples = set()
    for m in list_of_matrices:
        nz_tuples.update(zip(m.row, m.col))
    nz_tuples = sorted(nz_tuples)
    self._nz_tuples = nz_tuples
    self._row, self._col = list(zip(*nz_tuples))
    row_col_to_nz_map = {t: i for i, t in enumerate(nz_tuples)}
    self._shape = None
    self._maps = list()
    for m in list_of_matrices:
        nnz = len(m.data)
        map_row = np.zeros(nnz)
        map_col = np.zeros(nnz)
        for i in range(nnz):
            map_col[i] = i
            map_row[i] = row_col_to_nz_map[m.row[i], m.col[i]]
        mp = coo_matrix((np.ones(nnz), (map_row, map_col)), shape=(len(row_col_to_nz_map), nnz))
        self._maps.append(mp)
        if self._shape is None:
            self._shape = m.shape
        else:
            assert self._shape == m.shape