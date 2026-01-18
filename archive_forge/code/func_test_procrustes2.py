import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from pytest import raises as assert_raises
from scipy.spatial import procrustes
def test_procrustes2(self):
    m1, m3, disp13 = procrustes(self.data1, self.data3)
    m3_2, m1_2, disp31 = procrustes(self.data3, self.data1)
    assert_almost_equal(disp13, disp31)
    rand1 = np.array([[2.61955202, 0.30522265, 0.55515826], [0.41124708, -0.03966978, -0.31854548], [0.91910318, 1.39451809, -0.15295084], [2.00452023, 0.50150048, 0.29485268], [0.09453595, 0.67528885, 0.03283872], [0.07015232, 2.18892599, -1.67266852], [0.65029688, 1.60551637, 0.80013549], [-0.6607528, 0.53644208, 0.17033891]])
    rand3 = np.array([[0.0809969, 0.09731461, -0.173442], [-1.84888465, -0.92589646, -1.29335743], [0.67031855, -1.35957463, 0.41938621], [0.73967209, -0.20230757, 0.52418027], [0.17752796, 0.09065607, 0.29827466], [0.47999368, -0.88455717, -0.57547934], [-0.11486344, -0.12608506, -0.3395779], [-0.86106154, -0.28687488, 0.9644429]])
    res1, res3, disp13 = procrustes(rand1, rand3)
    res3_2, res1_2, disp31 = procrustes(rand3, rand1)
    assert_almost_equal(disp13, disp31)