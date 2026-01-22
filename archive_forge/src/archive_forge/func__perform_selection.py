import numpy as np
from .base import product
from .. import h5s, h5r, _selector
def _perform_selection(self, points, op):
    """ Internal method which actually performs the selection """
    points = np.asarray(points, order='C', dtype='u8')
    if len(points.shape) == 1:
        points.shape = (1, points.shape[0])
    if self._id.get_select_type() != h5s.SEL_POINTS:
        op = h5s.SELECT_SET
    if len(points) == 0:
        self._id.select_none()
    else:
        self._id.select_elements(points, op)