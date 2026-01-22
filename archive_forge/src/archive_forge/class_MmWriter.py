from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
class MmWriter:
    """Store a corpus in `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_,
    using :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    Notes
    -----
    The output is written one document at a time, not the whole matrix at once (unlike e.g. `scipy.io.mmread`).
    This allows you to write corpora which are larger than the available RAM.

    The output file is created in a single pass through the input corpus, so that the input can be
    a once-only stream (generator).

    To achieve this, a fake MM header is written first, corpus statistics are collected
    during the pass (shape of the matrix, number of non-zeroes), followed by a seek back to the beginning of the file,
    rewriting the fake header with the final values.

    """
    HEADER_LINE = b'%%MatrixMarket matrix coordinate real general\n'

    def __init__(self, fname):
        """

        Parameters
        ----------
        fname : str
            Path to output file.

        """
        self.fname = fname
        if fname.endswith('.gz') or fname.endswith('.bz2'):
            raise NotImplementedError('compressed output not supported with MmWriter')
        self.fout = utils.open(self.fname, 'wb+')
        self.headers_written = False

    def write_headers(self, num_docs, num_terms, num_nnz):
        """Write headers to file.

        Parameters
        ----------
        num_docs : int
            Number of documents in corpus.
        num_terms : int
            Number of term in corpus.
        num_nnz : int
            Number of non-zero elements in corpus.

        """
        self.fout.write(MmWriter.HEADER_LINE)
        if num_nnz < 0:
            logger.info('saving sparse matrix to %s', self.fname)
            self.fout.write(utils.to_utf8(' ' * 50 + '\n'))
        else:
            logger.info('saving sparse %sx%s matrix with %i non-zero entries to %s', num_docs, num_terms, num_nnz, self.fname)
            self.fout.write(utils.to_utf8('%s %s %s\n' % (num_docs, num_terms, num_nnz)))
        self.last_docno = -1
        self.headers_written = True

    def fake_headers(self, num_docs, num_terms, num_nnz):
        """Write "fake" headers to file, to be rewritten once we've scanned the entire corpus.

        Parameters
        ----------
        num_docs : int
            Number of documents in corpus.
        num_terms : int
            Number of term in corpus.
        num_nnz : int
            Number of non-zero elements in corpus.

        """
        stats = '%i %i %i' % (num_docs, num_terms, num_nnz)
        if len(stats) > 50:
            raise ValueError('Invalid stats: matrix too large!')
        self.fout.seek(len(MmWriter.HEADER_LINE))
        self.fout.write(utils.to_utf8(stats))

    def write_vector(self, docno, vector):
        """Write a single sparse vector to the file.

        Parameters
        ----------
        docno : int
            Number of document.
        vector : list of (int, number)
            Document in BoW format.

        Returns
        -------
        (int, int)
            Max word index in vector and len of vector. If vector is empty, return (-1, 0).

        """
        assert self.headers_written, 'must write Matrix Market file headers before writing data!'
        assert self.last_docno < docno, 'documents %i and %i not in sequential order!' % (self.last_docno, docno)
        vector = sorted(((i, w) for i, w in vector if abs(w) > 1e-12))
        for termid, weight in vector:
            self.fout.write(utils.to_utf8('%i %i %s\n' % (docno + 1, termid + 1, weight)))
        self.last_docno = docno
        return (vector[-1][0], len(vector)) if vector else (-1, 0)

    @staticmethod
    def write_corpus(fname, corpus, progress_cnt=1000, index=False, num_terms=None, metadata=False):
        """Save the corpus to disk in `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_.

        Parameters
        ----------
        fname : str
            Filename of the resulting file.
        corpus : iterable of list of (int, number)
            Corpus in streamed bag-of-words format.
        progress_cnt : int, optional
            Print progress for every `progress_cnt` number of documents.
        index : bool, optional
            Return offsets?
        num_terms : int, optional
            Number of terms in the corpus. If provided, the `corpus.num_terms` attribute (if any) will be ignored.
        metadata : bool, optional
            Generate a metadata file?

        Returns
        -------
        offsets : {list of int, None}
            List of offsets (if index=True) or nothing.

        Notes
        -----
        Documents are processed one at a time, so the whole corpus is allowed to be larger than the available RAM.

        See Also
        --------
        :func:`gensim.corpora.mmcorpus.MmCorpus.save_corpus`
            Save corpus to disk.

        """
        mw = MmWriter(fname)
        mw.write_headers(-1, -1, -1)
        _num_terms, num_nnz = (0, 0)
        docno, poslast = (-1, -1)
        offsets = []
        if hasattr(corpus, 'metadata'):
            orig_metadata = corpus.metadata
            corpus.metadata = metadata
            if metadata:
                docno2metadata = {}
        else:
            metadata = False
        for docno, doc in enumerate(corpus):
            if metadata:
                bow, data = doc
                docno2metadata[docno] = data
            else:
                bow = doc
            if docno % progress_cnt == 0:
                logger.info('PROGRESS: saving document #%i', docno)
            if index:
                posnow = mw.fout.tell()
                if posnow == poslast:
                    offsets[-1] = -1
                offsets.append(posnow)
                poslast = posnow
            max_id, veclen = mw.write_vector(docno, bow)
            _num_terms = max(_num_terms, 1 + max_id)
            num_nnz += veclen
        if metadata:
            utils.pickle(docno2metadata, fname + '.metadata.cpickle')
            corpus.metadata = orig_metadata
        num_docs = docno + 1
        num_terms = num_terms or _num_terms
        if num_docs * num_terms != 0:
            logger.info('saved %ix%i matrix, density=%.3f%% (%i/%i)', num_docs, num_terms, 100.0 * num_nnz / (num_docs * num_terms), num_nnz, num_docs * num_terms)
        mw.fake_headers(num_docs, num_terms, num_nnz)
        mw.close()
        if index:
            return offsets

    def __del__(self):
        """Close `self.fout` file. Alias for :meth:`~gensim.matutils.MmWriter.close`.

        Warnings
        --------
        Closing the file explicitly via the close() method is preferred and safer.

        """
        self.close()

    def close(self):
        """Close `self.fout` file."""
        logger.debug('closing %s', self.fname)
        if hasattr(self, 'fout'):
            self.fout.close()