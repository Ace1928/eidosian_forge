import datetime
import pickle
import sys
from copy import deepcopy
from tests.base import BaseTestCase
from pyasn1.type import useful
class GeneralizedTimePicklingTestCase(unittest.TestCase):

    def testSchemaPickling(self):
        old_asn1 = useful.GeneralizedTime()
        serialised = pickle.dumps(old_asn1)
        assert serialised
        new_asn1 = pickle.loads(serialised)
        assert type(new_asn1) == useful.GeneralizedTime
        assert old_asn1.isSameTypeWith(new_asn1)

    def testValuePickling(self):
        old_asn1 = useful.GeneralizedTime('20170916234254+0130')
        serialised = pickle.dumps(old_asn1)
        assert serialised
        new_asn1 = pickle.loads(serialised)
        assert new_asn1 == old_asn1