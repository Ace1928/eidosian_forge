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
class TweetViewer(TweetHandlerI):
    """
    Handle data by sending it to the terminal.
    """

    def handle(self, data):
        """
        Direct data to `sys.stdout`

        :return: return ``False`` if processing should cease, otherwise return ``True``.
        :rtype: bool
        :param data: Tweet object returned by Twitter API
        """
        text = data['text']
        print(text)
        self.check_date_limit(data)
        if self.do_stop:
            return

    def on_finish(self):
        print(f'Written {self.counter} Tweets')