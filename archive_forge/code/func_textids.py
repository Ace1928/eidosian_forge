from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def textids(self, fileids=None, categories=None):
    """
        In the pl196x corpus each category is stored in single
        file and thus both methods provide identical functionality. In order
        to accommodate finer granularity, a non-standard textids() method was
        implemented. All the main functions can be supplied with a list
        of required chunks---giving much more control to the user.
        """
    fileids, _ = self._resolve(fileids, categories)
    if fileids is None:
        return sorted(self._t2f)
    if isinstance(fileids, str):
        fileids = [fileids]
    return sorted(sum((self._f2t[d] for d in fileids), []))