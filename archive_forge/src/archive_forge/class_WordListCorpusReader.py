from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import line_tokenize
class WordListCorpusReader(CorpusReader):
    """
    List of words, one per line.  Blank lines are ignored.
    """

    def words(self, fileids=None, ignore_lines_startswith='\n'):
        return [line for line in line_tokenize(self.raw(fileids)) if not line.startswith(ignore_lines_startswith)]