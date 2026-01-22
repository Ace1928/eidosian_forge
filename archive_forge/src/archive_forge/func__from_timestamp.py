import datetime
import numbers
import abc
import bisect
import pytz
@staticmethod
def _from_timestamp(input):
    """
        If input is a real number, interpret it as a Unix timestamp
        (seconds sinc Epoch in UTC) and return a timezone-aware
        datetime object. Otherwise return input unchanged.
        """
    if not isinstance(input, numbers.Real):
        return input
    return from_timestamp(input)