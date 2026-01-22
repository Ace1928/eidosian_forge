from calendar import timegm
from decimal import Decimal as MyDecimal, ROUND_HALF_EVEN
from email.utils import formatdate
import six
from flask_restful import marshal
from flask import url_for, request
def _iso8601(dt):
    """Turn a datetime object into an ISO8601 formatted date.

    Example::

        fields._iso8601(datetime(2012, 1, 1, 0, 0)) => "2012-01-01T00:00:00"

    :param dt: The datetime to transform
    :type dt: datetime
    :return: A ISO 8601 formatted date string
    """
    return dt.isoformat()