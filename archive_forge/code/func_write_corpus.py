import logging
from collections import defaultdict
from gensim import utils
from gensim.corpora import Dictionary
from gensim.corpora import IndexedCorpus
from gensim.matutils import MmReader
from gensim.matutils import MmWriter
@staticmethod
def write_corpus(fname, corpus, progress_cnt=1000, index=False):
    """Write corpus in file.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus: iterable of list of (int, int)
            Corpus in BoW format.
        progress_cnt : int, optional
            Progress counter, write log message each `progress_cnt` documents.
        index : bool, optional
            If True - return offsets, otherwise - nothing.

        Return
        ------
        list of int
            Sequence of offsets to documents (in bytes), only if index=True.

        """
    writer = UciWriter(fname)
    writer.write_headers()
    num_terms, num_nnz = (0, 0)
    docno, poslast = (-1, -1)
    offsets = []
    for docno, bow in enumerate(corpus):
        if docno % progress_cnt == 0:
            logger.info('PROGRESS: saving document #%i', docno)
        if index:
            posnow = writer.fout.tell()
            if posnow == poslast:
                offsets[-1] = -1
            offsets.append(posnow)
            poslast = posnow
        vector = [(x, int(y)) for x, y in bow if int(y) != 0]
        max_id, veclen = writer.write_vector(docno, vector)
        num_terms = max(num_terms, 1 + max_id)
        num_nnz += veclen
    num_docs = docno + 1
    if num_docs * num_terms != 0:
        logger.info('saved %ix%i matrix, density=%.3f%% (%i/%i)', num_docs, num_terms, 100.0 * num_nnz / (num_docs * num_terms), num_nnz, num_docs * num_terms)
    writer.update_headers(num_docs, num_terms, num_nnz)
    writer.close()
    if index:
        return offsets