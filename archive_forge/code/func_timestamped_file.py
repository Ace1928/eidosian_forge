import datetime
import gzip
import itertools
import json
import os
import time
import requests
from twython import Twython, TwythonStreamer
from twython.exceptions import TwythonError, TwythonRateLimitError
from nltk.twitter.api import BasicTweetHandler, TweetHandlerI
from nltk.twitter.util import credsfromfile, guess_path
def timestamped_file(self):
    """
        :return: timestamped file name
        :rtype: str
        """
    subdir = self.subdir
    fprefix = self.fprefix
    if subdir:
        if not os.path.exists(subdir):
            os.mkdir(subdir)
    fname = os.path.join(subdir, fprefix)
    fmt = '%Y%m%d-%H%M%S'
    timestamp = datetime.datetime.now().strftime(fmt)
    if self.gzip_compress:
        suffix = '.gz'
    else:
        suffix = ''
    outfile = f'{fname}.{timestamp}.json{suffix}'
    return outfile