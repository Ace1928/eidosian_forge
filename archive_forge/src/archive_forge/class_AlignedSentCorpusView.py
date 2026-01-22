from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import (
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.translate import AlignedSent, Alignment
class AlignedSentCorpusView(StreamBackedCorpusView):
    """
    A specialized corpus view for aligned sentences.
    ``AlignedSentCorpusView`` objects are typically created by
    ``AlignedCorpusReader`` (not directly by nltk users).
    """

    def __init__(self, corpus_file, encoding, aligned, group_by_sent, word_tokenizer, sent_tokenizer, alignedsent_block_reader):
        self._aligned = aligned
        self._group_by_sent = group_by_sent
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._alignedsent_block_reader = alignedsent_block_reader
        StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

    def read_block(self, stream):
        block = [self._word_tokenizer.tokenize(sent_str) for alignedsent_str in self._alignedsent_block_reader(stream) for sent_str in self._sent_tokenizer.tokenize(alignedsent_str)]
        if self._aligned:
            block[2] = Alignment.fromstring(' '.join(block[2]))
            block = [AlignedSent(*block)]
        elif self._group_by_sent:
            block = [block[0]]
        else:
            block = block[0]
        return block