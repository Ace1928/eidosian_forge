import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
Determine if retry criteria matches.

        Note that either ``response`` is not None and ``caught_exception`` is
        None or ``response`` is None and ``caught_exception`` is not None.

        :type attempt_number: int
        :param attempt_number: The total number of times we've attempted
            to send the request.

        :param response: The HTTP response (if one was received).

        :type caught_exception: Exception
        :param caught_exception: Any exception that was caught while trying to
            send the HTTP response.

        :return: True, if the retry criteria matches (and therefore a retry
            should occur.  False if the criteria does not match.

        