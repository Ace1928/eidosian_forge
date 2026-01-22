from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
class StringCategoryCorpusReader(CorpusReader):

    def __init__(self, root, fileids, delimiter=' ', encoding='utf8'):
        """
        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        :param delimiter: Field delimiter
        """
        CorpusReader.__init__(self, root, fileids, encoding)
        self._delimiter = delimiter

    def tuples(self, fileids=None):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat([StreamBackedCorpusView(fileid, self._read_tuple_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def _read_tuple_block(self, stream):
        line = stream.readline().strip()
        if line:
            return [tuple(line.split(self._delimiter, 1))]
        else:
            return []