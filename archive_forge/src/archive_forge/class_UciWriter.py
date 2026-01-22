import logging
from collections import defaultdict
from gensim import utils
from gensim.corpora import Dictionary
from gensim.corpora import IndexedCorpus
from gensim.matutils import MmReader
from gensim.matutils import MmWriter
class UciWriter(MmWriter):
    """Writer of UCI format for :class:`gensim.corpora.ucicorpus.UciCorpus`.

    Notes
    ---------
    This corpus format is identical to `Matrix Market format<http://math.nist.gov/MatrixMarket/formats.html>,
    except for different file headers. There is no format line, and the first three lines of the file
    contain `number_docs`, `num_terms`, and `num_nnz`, one value per line.

    """
    MAX_HEADER_LENGTH = 20
    FAKE_HEADER = utils.to_utf8(' ' * MAX_HEADER_LENGTH + '\n')

    def write_headers(self):
        """Write blank header lines. Will be updated later, once corpus stats are known."""
        for _ in range(3):
            self.fout.write(self.FAKE_HEADER)
        self.last_docno = -1
        self.headers_written = True

    def update_headers(self, num_docs, num_terms, num_nnz):
        """Update headers with actual values."""
        offset = 0
        values = [utils.to_utf8(str(n)) for n in [num_docs, num_terms, num_nnz]]
        for value in values:
            if len(value) > len(self.FAKE_HEADER):
                raise ValueError('Invalid header: value too large!')
            self.fout.seek(offset)
            self.fout.write(value)
            offset += len(self.FAKE_HEADER)

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