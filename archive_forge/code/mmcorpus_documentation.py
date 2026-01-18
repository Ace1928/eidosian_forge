import logging
from gensim import matutils
from gensim.corpora import IndexedCorpus
Save a corpus to disk in the sparse coordinate Matrix Market format.

        Parameters
        ----------
        fname : str
            Path to file.
        corpus : iterable of list of (int, number)
            Corpus in Bow format.
        id2word : dict of (int, str), optional
            Mapping between word_id -> word. Used to retrieve the total vocabulary size if provided.
            Otherwise, the total vocabulary size is estimated based on the highest feature id encountered in `corpus`.
        progress_cnt : int, optional
            How often to report (log) progress.
        metadata : bool, optional
            Writes out additional metadata?

        Warnings
        --------
        This function is automatically called by :class:`~gensim.corpora.mmcorpus.MmCorpus.serialize`, don't
        call it directly, call :class:`~gensim.corpora.mmcorpus.MmCorpus.serialize` instead.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.corpora.mmcorpus import MmCorpus
            >>> from gensim.test.utils import datapath
            >>>
            >>> corpus = MmCorpus(datapath('test_mmcorpus_with_index.mm'))
            >>>
            >>> MmCorpus.save_corpus("random", corpus)  # Do not do it, use `serialize` instead.
            [97, 121, 169, 201, 225, 249, 258, 276, 303]

        