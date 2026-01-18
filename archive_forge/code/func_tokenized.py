import json
import os
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, ZipFilePathPointer, concat
from nltk.tokenize import TweetTokenizer
def tokenized(self, fileids=None):
    """
        :return: the given file(s) as a list of the text content of Tweets as
            as a list of words, screenanames, hashtags, URLs and punctuation symbols.

        :rtype: list(list(str))
        """
    tweets = self.strings(fileids)
    tokenizer = self._word_tokenizer
    return [tokenizer.tokenize(t) for t in tweets]