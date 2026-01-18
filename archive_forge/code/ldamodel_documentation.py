import logging
import numbers
import os
import time
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from gensim import interfaces, utils, matutils
from gensim.matutils import (
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback
Load a previously saved :class:`gensim.models.ldamodel.LdaModel` from file.

        See Also
        --------
        :meth:`~gensim.models.ldamodel.LdaModel.save`
            Save model.

        Parameters
        ----------
        fname : str
            Path to the file where the model is stored.
        *args
            Positional arguments propagated to :meth:`~gensim.utils.SaveLoad.load`.
        **kwargs
            Key word arguments propagated to :meth:`~gensim.utils.SaveLoad.load`.

        Examples
        --------
        Large arrays can be memmap'ed back as read-only (shared memory) by setting `mmap='r'`:

        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> fname = datapath("lda_3_0_1_model")
            >>> lda = LdaModel.load(fname, mmap='r')

        