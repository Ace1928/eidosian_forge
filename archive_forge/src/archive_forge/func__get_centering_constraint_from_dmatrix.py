import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def _get_centering_constraint_from_dmatrix(design_matrix):
    """ Computes the centering constraint from the given design matrix.

    We want to ensure that if ``b`` is the array of parameters, our
    model is centered, ie ``np.mean(np.dot(design_matrix, b))`` is zero.
    We can rewrite this as ``np.dot(c, b)`` being zero with ``c`` a 1-row
    constraint matrix containing the mean of each column of ``design_matrix``.

    :param design_matrix: The 2-d array design matrix.
    :return: A 2-d array (1 x ncols(design_matrix)) defining the
     centering constraint.
    """
    return design_matrix.mean(axis=0).reshape((1, design_matrix.shape[1]))