import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_use_set_time_helpers(self):
    dt = datetime.datetime.now()
    document1 = ProvDocument()
    document1.activity(EX_NS['a8'], startTime=dt, endTime=dt)
    document2 = ProvDocument()
    a = document2.activity(EX_NS['a8'])
    a.set_time(startTime=dt, endTime=dt)
    self.assertEqual(document1, document2)
    self.assertEqual(a.get_startTime(), dt)
    self.assertEqual(a.get_endTime(), dt)