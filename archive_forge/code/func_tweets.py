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
def tweets(self, keywords='', follow='', to_screen=True, stream=True, limit=100, date_limit=None, lang='en', repeat=False, gzip_compress=False):
    """
        Process some Tweets in a simple manner.

        :param str keywords: Keywords to use for searching or filtering
        :param list follow: UserIDs to use for filtering Tweets from the public stream
        :param bool to_screen: If `True`, display the tweet texts on the screen,            otherwise print to a file

        :param bool stream: If `True`, use the live public stream,            otherwise search past public Tweets

        :param int limit: The number of data items to process in the current            round of processing.

        :param tuple date_limit: The date at which to stop collecting            new data. This should be entered as a tuple which can serve as the            argument to `datetime.datetime`.            E.g. `date_limit=(2015, 4, 1, 12, 40)` for 12:30 pm on April 1 2015.
            Note that, in the case of streaming, this is the maximum date, i.e.            a date in the future; if not, it is the minimum date, i.e. a date            in the past

        :param str lang: language

        :param bool repeat: A flag to determine whether multiple files should            be written. If `True`, the length of each file will be set by the            value of `limit`. Use only if `to_screen` is `False`. See also
            :py:func:`handle`.

        :param gzip_compress: if `True`, output files are compressed with gzip.
        """
    if stream:
        upper_date_limit = date_limit
        lower_date_limit = None
    else:
        upper_date_limit = None
        lower_date_limit = date_limit
    if to_screen:
        handler = TweetViewer(limit=limit, upper_date_limit=upper_date_limit, lower_date_limit=lower_date_limit)
    else:
        handler = TweetWriter(limit=limit, upper_date_limit=upper_date_limit, lower_date_limit=lower_date_limit, repeat=repeat, gzip_compress=gzip_compress)
    if to_screen:
        handler = TweetViewer(limit=limit)
    else:
        if stream:
            upper_date_limit = date_limit
            lower_date_limit = None
        else:
            upper_date_limit = None
            lower_date_limit = date_limit
        handler = TweetWriter(limit=limit, upper_date_limit=upper_date_limit, lower_date_limit=lower_date_limit, repeat=repeat, gzip_compress=gzip_compress)
    if stream:
        self.streamer.register(handler)
        if keywords == '' and follow == '':
            self.streamer.sample()
        else:
            self.streamer.filter(track=keywords, follow=follow, lang=lang)
    else:
        self.query.register(handler)
        if keywords == '':
            raise ValueError('Please supply at least one keyword to search for.')
        else:
            self.query._search_tweets(keywords, limit=limit, lang=lang)