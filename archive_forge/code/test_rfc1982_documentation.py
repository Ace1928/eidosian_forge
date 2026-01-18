import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest

        The pairs of values 0 and 128, 1 and 129, 2 and 130, etc, to 127 and 255
        are not equal, but in each pair, neither number is defined as being
        greater than, or less than, the other.
        