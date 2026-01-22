from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.parse import DependencyGraph
from nltk.tokenize import *
class DependencyCorpusReader(SyntaxCorpusReader):

    def __init__(self, root, fileids, encoding='utf8', word_tokenizer=TabTokenizer(), sent_tokenizer=RegexpTokenizer('\n', gaps=True), para_block_reader=read_blankline_block):
        SyntaxCorpusReader.__init__(self, root, fileids, encoding)

    def words(self, fileids=None):
        return concat([DependencyCorpusView(fileid, False, False, False, encoding=enc) for fileid, enc in self.abspaths(fileids, include_encoding=True)])

    def tagged_words(self, fileids=None):
        return concat([DependencyCorpusView(fileid, True, False, False, encoding=enc) for fileid, enc in self.abspaths(fileids, include_encoding=True)])

    def sents(self, fileids=None):
        return concat([DependencyCorpusView(fileid, False, True, False, encoding=enc) for fileid, enc in self.abspaths(fileids, include_encoding=True)])

    def tagged_sents(self, fileids=None):
        return concat([DependencyCorpusView(fileid, True, True, False, encoding=enc) for fileid, enc in self.abspaths(fileids, include_encoding=True)])

    def parsed_sents(self, fileids=None):
        sents = concat([DependencyCorpusView(fileid, False, True, True, encoding=enc) for fileid, enc in self.abspaths(fileids, include_encoding=True)])
        return [DependencyGraph(sent) for sent in sents]