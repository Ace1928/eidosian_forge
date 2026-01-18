import os
from ... import recordio, ndarray
def transform_first(self, fn, lazy=True):
    """Returns a new dataset with the first element of each sample
        transformed by the transformer function `fn`.

        This is useful, for example, when you only want to transform data
        while keeping label as is.

        Parameters
        ----------
        fn : callable
            A transformer function that takes the first elemtn of a sample
            as input and returns the transformed element.
        lazy : bool, default True
            If False, transforms all samples at once. Otherwise,
            transforms each sample on demand. Note that if `fn`
            is stochastic, you must set lazy to True or you will
            get the same result on all epochs.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
    return self.transform(_TransformFirstClosure(fn), lazy)