import warnings
from collections import OrderedDict
import numpy as np
from gensim import utils
Translate the target model's document vector to the source model's document vector

        Parameters
        ----------
        target_doc_vec : numpy.ndarray
            Document vector from the target document, whose document are not in the source model.

        Returns
        -------
        numpy.ndarray
            Vector `target_doc_vec` in the source model.

        