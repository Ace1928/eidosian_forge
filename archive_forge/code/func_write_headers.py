import logging
from collections import defaultdict
from gensim import utils
from gensim.corpora import Dictionary
from gensim.corpora import IndexedCorpus
from gensim.matutils import MmReader
from gensim.matutils import MmWriter
def write_headers(self):
    """Write blank header lines. Will be updated later, once corpus stats are known."""
    for _ in range(3):
        self.fout.write(self.FAKE_HEADER)
    self.last_docno = -1
    self.headers_written = True