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
def user_info_from_id(self, userids):
    """
        Convert a list of userIDs into a variety of information about the users.

        See <https://dev.twitter.com/rest/reference/get/users/show>.

        :param list userids: A list of integer strings corresponding to Twitter userIDs
        :rtype: list(json)
        """
    return [self.show_user(user_id=userid) for userid in userids]