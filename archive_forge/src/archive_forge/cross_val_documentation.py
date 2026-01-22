from statsmodels.compat.python import lrange
import numpy as np
from itertools import combinations

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
        