import datetime
import pickle
import sys
from copy import deepcopy
from tests.base import BaseTestCase
from pyasn1.type import useful
class UTCTimeTestCase(BaseTestCase):

    def testFromDateTime(self):
        assert useful.UTCTime.fromDateTime(datetime.datetime(2017, 7, 11, 0, 1, 2, tzinfo=UTC)) == '170711000102Z'

    def testToDateTime0(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2) == useful.UTCTime('170711000102').asDateTime

    def testToDateTime1(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2, tzinfo=UTC) == useful.UTCTime('170711000102Z').asDateTime

    def testToDateTime2(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2, tzinfo=UTC) == useful.UTCTime('170711000102+0000').asDateTime

    def testToDateTime3(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2, tzinfo=UTC2) == useful.UTCTime('170711000102+0200').asDateTime

    def testToDateTime4(self):
        assert datetime.datetime(2017, 7, 11, 0, 1) == useful.UTCTime('1707110001').asDateTime