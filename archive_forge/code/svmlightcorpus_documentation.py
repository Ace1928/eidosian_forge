from __future__ import with_statement
import logging
from gensim import utils
from gensim.corpora import IndexedCorpus
Convert BoW representation of document in SVMlight format.
        This method inverse of :meth:`~gensim.corpora.svmlightcorpus.SvmLightCorpus.line2doc`.

        Parameters
        ----------
        doc : list of (int, float)
            Document in BoW format.
        label : int, optional
            Document label (if provided).

        Returns
        -------
        str
            `doc` in SVMlight format.

        