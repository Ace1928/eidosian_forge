import numbers
from . import _cluster  # type: ignore
def __check_distancematrix(distancematrix):
    if distancematrix is None:
        return distancematrix
    if isinstance(distancematrix, np.ndarray):
        distancematrix = np.require(distancematrix, dtype='d', requirements='C')
    else:
        try:
            distancematrix = np.array(distancematrix, dtype='d')
        except ValueError:
            n = len(distancematrix)
            d = [None] * n
            for i, row in enumerate(distancematrix):
                if isinstance(row, np.ndarray):
                    row = np.require(row, dtype='d', requirements='C')
                else:
                    row = np.array(row, dtype='d')
                if row.ndim != 1:
                    raise ValueError('row %d is not one-dimensional' % i) from None
                m = len(row)
                if m != i:
                    raise ValueError('row %d has incorrect size (%d, expected %d)' % (i, m, i)) from None
                if np.isnan(row).any():
                    raise ValueError('distancematrix contains NaN values') from None
                d[i] = row
            return d
    if np.isnan(distancematrix).any():
        raise ValueError('distancematrix contains NaN values')
    return distancematrix