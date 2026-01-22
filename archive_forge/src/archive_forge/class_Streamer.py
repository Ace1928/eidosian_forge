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
class Streamer(TwythonStreamer):
    """
    Retrieve data from the Twitter Streaming API.

    The streaming API requires
    `OAuth 1.0 <https://en.wikipedia.org/wiki/OAuth>`_ authentication.
    """

    def __init__(self, app_key, app_secret, oauth_token, oauth_token_secret):
        self.handler = None
        self.do_continue = True
        TwythonStreamer.__init__(self, app_key, app_secret, oauth_token, oauth_token_secret)

    def register(self, handler):
        """
        Register a method for handling Tweets.

        :param TweetHandlerI handler: method for viewing
        """
        self.handler = handler

    def on_success(self, data):
        """
        :param data: response from Twitter API
        """
        if self.do_continue:
            if self.handler is not None:
                if 'text' in data:
                    self.handler.counter += 1
                    self.handler.handle(data)
                    self.do_continue = self.handler.do_continue()
            else:
                raise ValueError('No data handler has been registered.')
        else:
            self.disconnect()
            self.handler.on_finish()

    def on_error(self, status_code, data):
        """
        :param status_code: The status code returned by the Twitter API
        :param data: The response from Twitter API

        """
        print(status_code)

    def sample(self):
        """
        Wrapper for 'statuses / sample' API call
        """
        while self.do_continue:
            try:
                self.statuses.sample()
            except requests.exceptions.ChunkedEncodingError as e:
                if e is not None:
                    print(f'Error (stream will continue): {e}')
                continue

    def filter(self, track='', follow='', lang='en'):
        """
        Wrapper for 'statuses / filter' API call
        """
        while self.do_continue:
            try:
                if track == '' and follow == '':
                    msg = "Please supply a value for 'track', 'follow'"
                    raise ValueError(msg)
                self.statuses.filter(track=track, follow=follow, lang=lang)
            except requests.exceptions.ChunkedEncodingError as e:
                if e is not None:
                    print(f'Error (stream will continue): {e}')
                continue