import codecs
from xml.etree import ElementTree
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import *
from nltk.data import SeekableUnicodeStreamReader
from nltk.internals import ElementWrapper
from nltk.tokenize import WordPunctTokenizer
class XMLCorpusReader(CorpusReader):
    """
    Corpus reader for corpora whose documents are xml files.

    Note that the ``XMLCorpusReader`` constructor does not take an
    ``encoding`` argument, because the unicode encoding is specified by
    the XML files themselves.  See the XML specs for more info.
    """

    def __init__(self, root, fileids, wrap_etree=False):
        self._wrap_etree = wrap_etree
        CorpusReader.__init__(self, root, fileids)

    def xml(self, fileid=None):
        if fileid is None and len(self._fileids) == 1:
            fileid = self._fileids[0]
        if not isinstance(fileid, str):
            raise TypeError('Expected a single file identifier string')
        with self.abspath(fileid).open() as fp:
            elt = ElementTree.parse(fp).getroot()
        if self._wrap_etree:
            elt = ElementWrapper(elt)
        return elt

    def words(self, fileid=None):
        """
        Returns all of the words and punctuation symbols in the specified file
        that were in text nodes -- ie, tags are ignored. Like the xml() method,
        fileid can only specify one file.

        :return: the given file's text nodes as a list of words and punctuation symbols
        :rtype: list(str)
        """
        elt = self.xml(fileid)
        encoding = self.encoding(fileid)
        word_tokenizer = WordPunctTokenizer()
        try:
            iterator = elt.getiterator()
        except:
            iterator = elt.iter()
        out = []
        for node in iterator:
            text = node.text
            if text is not None:
                if isinstance(text, bytes):
                    text = text.decode(encoding)
                toks = word_tokenizer.tokenize(text)
                out.extend(toks)
        return out