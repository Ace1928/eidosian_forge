import logging
import numpy
from gensim import interfaces, utils
Get document by `docno` index.

        Parameters
        ----------
        docno : {int, iterable of int}
            Document number or iterable of numbers (like a list of str).

        Returns
        -------
        list of (int, float)
            If `docno` is int - return document in BoW format.

        :class:`~gensim.utils.SlicedCorpus`
            If `docno` is iterable of int - return several documents in BoW format
            wrapped to :class:`~gensim.utils.SlicedCorpus`.

        Raises
        ------
        RuntimeError
            If index isn't exist.

        