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
def _search_tweets(self, keywords, limit=100, lang='en'):
    """
        Assumes that the handler has been informed. Fetches Tweets from
        search_tweets generator output and passses them to handler

        :param str keywords: A list of query terms to search for, written as        a comma-separated string.
        :param int limit: Number of Tweets to process
        :param str lang: language
        """
    while True:
        tweets = self.search_tweets(keywords=keywords, limit=limit, lang=lang, max_id=self.handler.max_id)
        for tweet in tweets:
            self.handler.handle(tweet)
        if not (self.handler.do_continue() and self.handler.repeat):
            break
    self.handler.on_finish()