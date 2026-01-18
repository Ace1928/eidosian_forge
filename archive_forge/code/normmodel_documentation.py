import logging
from gensim import interfaces, matutils
Call the :func:`~gensim.models.normmodel.NormModel.normalize`.

        Parameters
        ----------
        bow : list of (int, number)
            Document in BoW format.

        Returns
        -------
        list of (int, number)
            Normalized document.

        