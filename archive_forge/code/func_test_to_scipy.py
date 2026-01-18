import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sp
from scipy.sparse import coo_matrix, bmat
from pyomo.contrib.pynumero.sparse import (
import warnings
def test_to_scipy(self):
    block = self.block_m
    m = self.basic_m
    scipy_mat = bmat([[block, block], [None, block]], format='coo')
    dinopy_mat = m.tocoo()
    drow = np.sort(dinopy_mat.row)
    dcol = np.sort(dinopy_mat.col)
    ddata = np.sort(dinopy_mat.data)
    srow = np.sort(scipy_mat.row)
    scol = np.sort(scipy_mat.col)
    sdata = np.sort(scipy_mat.data)
    self.assertListEqual(drow.tolist(), srow.tolist())
    self.assertListEqual(dcol.tolist(), scol.tolist())
    self.assertListEqual(ddata.tolist(), sdata.tolist())