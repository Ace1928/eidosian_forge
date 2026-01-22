import time as _time
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta, timezone, tzinfo
class TweetHandlerI(BasicTweetHandler):
    """
    Interface class whose subclasses should implement a handle method that
    Twitter clients can delegate to.
    """

    def __init__(self, limit=20, upper_date_limit=None, lower_date_limit=None):
        """
        :param int limit: The number of data items to process in the current        round of processing.

        :param tuple upper_date_limit: The date at which to stop collecting        new data. This should be entered as a tuple which can serve as the        argument to `datetime.datetime`.        E.g. `date_limit=(2015, 4, 1, 12, 40)` for 12:30 pm on April 1 2015.

        :param tuple lower_date_limit: The date at which to stop collecting        new data. See `upper_data_limit` for formatting.
        """
        BasicTweetHandler.__init__(self, limit)
        self.upper_date_limit = None
        self.lower_date_limit = None
        if upper_date_limit:
            self.upper_date_limit = datetime(*upper_date_limit, tzinfo=LOCAL)
        if lower_date_limit:
            self.lower_date_limit = datetime(*lower_date_limit, tzinfo=LOCAL)
        self.startingup = True

    @abstractmethod
    def handle(self, data):
        """
        Deal appropriately with data returned by the Twitter API
        """

    @abstractmethod
    def on_finish(self):
        """
        Actions when the tweet limit has been reached
        """

    def check_date_limit(self, data, verbose=False):
        """
        Validate date limits.
        """
        if self.upper_date_limit or self.lower_date_limit:
            date_fmt = '%a %b %d %H:%M:%S +0000 %Y'
            tweet_date = datetime.strptime(data['created_at'], date_fmt).replace(tzinfo=timezone.utc)
            if self.upper_date_limit and tweet_date > self.upper_date_limit or (self.lower_date_limit and tweet_date < self.lower_date_limit):
                if self.upper_date_limit:
                    message = 'earlier'
                    date_limit = self.upper_date_limit
                else:
                    message = 'later'
                    date_limit = self.lower_date_limit
                if verbose:
                    print('Date limit {} is {} than date of current tweet {}'.format(date_limit, message, tweet_date))
                self.do_stop = True