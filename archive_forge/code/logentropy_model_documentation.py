import logging
import math
from gensim import interfaces, matutils, utils
Get log entropy representation of the input vector and/or corpus.

        Parameters
        ----------
        bow : list of (int, int)
            Document in BoW format.

        Returns
        -------
        list of (int, float)
            Log-entropy vector for passed `bow`.

        