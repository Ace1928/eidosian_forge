from statsmodels.compat.python import lrange
import numpy as np
from itertools import combinations
class KStepAhead:
    """
    KStepAhead cross validation iterator:
    Provides fit/test indexes to split data in sequential sets
    """

    def __init__(self, n, k=1, start=None, kall=True, return_slice=True):
        """
        KStepAhead cross validation iterator:
        Provides train/test indexes to split data in train test sets

        Parameters
        ----------
        n: int
            Total number of elements
        k : int
            number of steps ahead
        start : int
            initial size of data for fitting
        kall : bool
            if true. all values for up to k-step ahead are included in the test index.
            If false, then only the k-th step ahead value is returnd


        Notes
        -----
        I do not think this is really useful, because it can be done with
        a very simple loop instead.
        Useful as a plugin, but it could return slices instead for faster array access.

        Examples
        --------
        >>> from scikits.learn import cross_val
        >>> X = [[1, 2], [3, 4]]
        >>> y = [1, 2]
        >>> loo = cross_val.LeaveOneOut(2)
        >>> for train_index, test_index in loo:
        ...    print "TRAIN:", train_index, "TEST:", test_index
        ...    X_train, X_test, y_train, y_test = cross_val.split(train_index, test_index, X, y)
        ...    print X_train, X_test, y_train, y_test
        TRAIN: [False  True] TEST: [ True False]
        [[3 4]] [[1 2]] [2] [1]
        TRAIN: [ True False] TEST: [False  True]
        [[1 2]] [[3 4]] [1] [2]
        """
        self.n = n
        self.k = k
        if start is None:
            start = int(np.trunc(n * 0.25))
        self.start = start
        self.kall = kall
        self.return_slice = return_slice

    def __iter__(self):
        n = self.n
        k = self.k
        start = self.start
        if self.return_slice:
            for i in range(start, n - k):
                train_slice = slice(None, i, None)
                if self.kall:
                    test_slice = slice(i, i + k)
                else:
                    test_slice = slice(i + k - 1, i + k)
                yield (train_slice, test_slice)
        else:
            for i in range(start, n - k):
                train_index = np.zeros(n, dtype=bool)
                train_index[:i] = True
                test_index = np.zeros(n, dtype=bool)
                if self.kall:
                    test_index[i:i + k] = True
                else:
                    test_index[i + k - 1:i + k] = True
                yield (train_index, test_index)

    def __repr__(self):
        return '%s.%s(n=%i)' % (self.__class__.__module__, self.__class__.__name__, self.n)