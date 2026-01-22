import json
import os
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, ZipFilePathPointer, concat
from nltk.tokenize import TweetTokenizer
def _read_tweets(self, stream):
    """
        Assumes that each line in ``stream`` is a JSON-serialised object.
        """
    tweets = []
    for i in range(10):
        line = stream.readline()
        if not line:
            return tweets
        tweet = json.loads(line)
        tweets.append(tweet)
    return tweets