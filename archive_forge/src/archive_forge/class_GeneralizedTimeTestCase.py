import datetime
import pickle
import sys
from copy import deepcopy
from tests.base import BaseTestCase
from pyasn1.type import useful
class GeneralizedTimeTestCase(BaseTestCase):

    def testFromDateTime(self):
        assert useful.GeneralizedTime.fromDateTime(datetime.datetime(2017, 7, 11, 0, 1, 2, 30000, tzinfo=UTC)) == '20170711000102.3Z'

    def testToDateTime0(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2) == useful.GeneralizedTime('20170711000102').asDateTime

    def testToDateTime1(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2, tzinfo=UTC) == useful.GeneralizedTime('20170711000102Z').asDateTime

    def testToDateTime2(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2, 30000, tzinfo=UTC) == useful.GeneralizedTime('20170711000102.3Z').asDateTime

    def testToDateTime3(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2, 30000, tzinfo=UTC) == useful.GeneralizedTime('20170711000102,3Z').asDateTime

    def testToDateTime4(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2, 30000, tzinfo=UTC) == useful.GeneralizedTime('20170711000102.3+0000').asDateTime

    def testToDateTime5(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2, 30000, tzinfo=UTC2) == useful.GeneralizedTime('20170711000102.3+0200').asDateTime

    def testToDateTime6(self):
        assert datetime.datetime(2017, 7, 11, 0, 1, 2, 30000, tzinfo=UTC2) == useful.GeneralizedTime('20170711000102.3+02').asDateTime

    def testToDateTime7(self):
        assert datetime.datetime(2017, 7, 11, 0, 1) == useful.GeneralizedTime('201707110001').asDateTime

    def testToDateTime8(self):
        assert datetime.datetime(2017, 7, 11, 0) == useful.GeneralizedTime('2017071100').asDateTime

    def testCopy(self):
        dt = useful.GeneralizedTime('20170916234254+0130').asDateTime
        assert dt == deepcopy(dt)